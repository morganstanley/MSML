import os.path as osp
from collections import OrderedDict
import math

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.cuda.amp import GradScaler, autocast

from dassl.engine import TRAINER_REGISTRY, TrainerX
from dassl.metrics import compute_accuracy
from dassl.utils import load_pretrained_weights, load_checkpoint
from dassl.optim import build_optimizer, build_lr_scheduler

from clip import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer

_tokenizer = _Tokenizer()


def load_clip_to_cpu(cfg):
    backbone_name = cfg.MODEL.BACKBONE.NAME
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url)

    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")
    design_details = {"trainer": 'CoCoOp',
                      "vision_depth": 0,
                      "language_depth": 0, "vision_ctx": 0,
                      "language_ctx": 0}
    model = clip.build_model(state_dict or model.state_dict(), design_details)

    return model


class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        
        self.text_projection = clip_model.text_projection
        # print(clip_model.text_projection)
        # self.text_projection = nn.Parameter(clip_model.text_projection.clone())
        
        
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection

        return x


# class TextEncoder(nn.Module):
#     def __init__(self, clip_model):
#         super().__init__()
#         self.transformer = clip_model.transformer
#         self.positional_embedding = clip_model.positional_embedding
#         self.ln_final = clip_model.ln_final
#         self.text_projection = clip_model.text_projection
#         self.dtype = clip_model.dtype

#     def forward(self, prompts, tokenized_prompts, compound_prompts_deeper_text):
#         x = prompts + self.positional_embedding.type(self.dtype)
#         x = x.permute(1, 0, 2)  # NLD -> LND
#         # Pass as the list, as nn.sequential cannot process multiple arguments in the forward pass
#         combined = [x, compound_prompts_deeper_text, 0]  # third argument is the counter which denotes depth of prompt
#         outputs = self.transformer(combined)
#         x = outputs[0]  # extract the x back from here
#         x = x.permute(1, 0, 2)  # LND -> NLD
#         x = self.ln_final(x).type(self.dtype)

#         # x.shape = [batch_size, n_ctx, transformer.width]
#         # take features from the eot embedding (eot_token is the highest number in each sequence)
#         x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection

#         return x

class fuser(nn.Module):
    def __init__(self,clip_model):
        super().__init__()
        
        self.expand = 512
        self.fuse1 = nn.Linear(512, self.expand).to(clip_model.dtype)
        self.fuse2 = nn.Linear(512, self.expand).to(clip_model.dtype)
        # with torch.no_grad():
#             self.fuse1.weight.copy_(torch.eye(512).to(clip_model.dtype))  # Identity matrix
#             self.fuse1.bias.zero_()
            
#             self.fuse2.weight.copy_(torch.eye(512).to(clip_model.dtype))  # Identity matrix
#             self.fuse2.bias.zero_()
        with torch.no_grad():
            # Initialize weights (e.g., Xavier uniform initialization)
            nn.init.xavier_uniform_(self.fuse1.weight)  # Example: Xavier uniform initialization
            # print(self.fuse1.weight.shape)
            self.fuse1.weight[:, :] = torch.eye( self.expand,512 )#[:512] 

            # Set bias to zero
            self.fuse1.bias.zero_()
            
            
            nn.init.xavier_uniform_(self.fuse2.weight)  # Example: Xavier uniform initialization
            self.fuse2.weight[:, :] = torch.eye( self.expand,512 )#[:512] 

            # Set bias to zero
            self.fuse2.bias.zero_()

        
        
    def forward(self, x,y):
        
        x = self.fuse1(x)
        y = self.fuse2(y)
        return x,y

class PromptLearner(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        n_cls = len(classnames)
        

        
        
        # self.add_n_cls = n_cls
        
        
        n_ctx = cfg.TRAINER.COCOOP.N_CTX
        ctx_init = cfg.TRAINER.COCOOP.CTX_INIT
        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]
        vis_dim = clip_model.visual.output_dim
        clip_imsize = clip_model.visual.input_resolution
        cfg_imsize = cfg.INPUT.SIZE[0]
        assert cfg_imsize == clip_imsize, f"cfg_imsize ({cfg_imsize}) must equal to clip_imsize ({clip_imsize})"

        if ctx_init:
            # use given words to initialize context vectors
            ctx_init = ctx_init.replace("_", " ")
            n_ctx = len(ctx_init.split(" "))
            prompt = clip.tokenize(ctx_init)
            with torch.no_grad():
                embedding = clip_model.token_embedding(prompt).type(dtype)
            ctx_vectors = embedding[0, 1: 1 + n_ctx, :]
            prompt_prefix = ctx_init
        else:
            # random initialization
            ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
            nn.init.normal_(ctx_vectors, std=0.02)
            prompt_prefix = " ".join(["X"] * n_ctx)

        print(f'Initial context: "{prompt_prefix}"')
        print(f"Number of context words (tokens): {n_ctx}")

        self.ctx = nn.Parameter(ctx_vectors)

#         self.meta_net = nn.Sequential(OrderedDict([
#             ("linear1", nn.Linear(vis_dim, vis_dim // 16)),
#             ("relu", nn.ReLU(inplace=True)),
#             ("linear2", nn.Linear(vis_dim // 16, ctx_dim))
#         ]))

#         if cfg.TRAINER.COCOOP.PREC == "fp16":
#             self.meta_net.half()

        classnames = [name.replace("_", " ") for name in classnames]
        
        # print(classnames[:10])
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        prompts = [prompt_prefix + " " + name + "." for name in classnames]

        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])  # (n_cls, n_tkn)
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)
        
        # self.prompts = nn.Parameter(embedding)
        # These token vectors will be saved when in save_model(),
        # but they should be ignored in load_model() as we want to use
        # those computed using the current class names
        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx:, :])  # CLS, EOS

        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.tokenized_prompts = tokenized_prompts  # torch.Tensor
        self.name_lens = name_lens

    def construct_prompts(self, ctx, prefix, suffix, label=None):
        # dim0 is either batch_size (during training) or n_cls (during testing)
        # ctx: context tokens, with shape of (dim0, n_ctx, ctx_dim)
        # prefix: the sos token, with shape of (n_cls, 1, ctx_dim)
        # suffix: remaining tokens, with shape of (n_cls, *, ctx_dim)

        if label is not None:
            prefix = prefix[label]
            suffix = suffix[label]

        prompts = torch.cat(
            [
                prefix,  # (dim0, 1, dim)
                ctx,  # (dim0, n_ctx, dim)
                suffix,  # (dim0, *, dim)
            ],
            dim=1,
        )

        return prompts

    def forward(self, im_features=None):
        prefix = self.token_prefix
        suffix = self.token_suffix
        ctx = self.ctx  # (n_ctx, ctx_dim)
        
        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)
        
#         #bias = self.meta_net(im_features)  # (batch, ctx_dim)
#         #bias = bias.unsqueeze(1)  # (batch, 1, ctx_dim)
        prompts = self.construct_prompts(ctx, prefix, suffix)
#         # add_prompts = torch.randn(self.add_n_cls,prompts.shape[1],prompts.shape[2]).to(prompts.dtype).to(prompts.device)
#         # prompts = torch.cat([prompts,add_prompts])
        # prompts = self.prompts
        return prompts
import torch

def compute_nc2(M):
    """
    Compute NC2 for a given matrix M with shape (K, d).

    Args:
        M (torch.Tensor): Input matrix of shape (K, d).

    Returns:
        torch.Tensor: The computed NC2 value (scalar).
    """
    # Compute MM^T
    MMT = torch.matmul(M, M.T)

    # Normalize MM^T by its Frobenius norm
    MMT_norm = torch.linalg.norm(MMT, ord='fro')
    normalized_MMT = MMT / MMT_norm

    # Get the number of rows (K) from M
    K = M.shape[0]

    # Compute the centering matrix
    I_K = torch.eye(K, device=M.device)  # Identity matrix of size K
    ones_K = torch.ones(K, 1, device=M.device)
    centering_matrix = I_K - (1 / K) * torch.matmul(ones_K, ones_K.T)

    # Compute the second term
    second_term = (1 / (K - 1)**0.5) * centering_matrix

    # Compute the Frobenius norm of the difference
    nc2 = torch.linalg.norm(normalized_MMT - second_term, ord='fro')

    return nc2


def find_orthogonal_min_sim_batch(x1_batch, basis_matrix):
    """
    Find vectors x2 for a batch of input vectors x1, such that x2 is orthogonal to a given basis 
    and minimizes similarity with x1.

    Parameters:
    x1_batch (torch.Tensor): The batch of input vectors x1 (shape: [batch_size, dim]).
    basis (list of torch.Tensor): A list of orthogonal basis vectors (1D tensors, shape: [dim]).

    Returns:
    torch.Tensor: The batch of vectors x2 (shape: [batch_size, dim]).
    """
    # Ensure basis vectors are PyTorch tensors
    # basis = [torch.tensor(e, dtype=x1_batch.dtype, device=x1_batch.device) for e in basis]
    # basis_matrix = torch.stack(basis)  # Shape: [num_basis, dim]

    # Compute the projection of each vector in x1_batch onto the basis
    # x1_batch has shape [batch_size, dim], basis_matrix has shape [num_basis, dim]
    projections = torch.einsum('bd,nd->bn', x1_batch, basis_matrix)  # Shape: [batch_size, num_basis]
    projections = projections.unsqueeze(-1) * basis_matrix  # Shape: [batch_size, num_basis, dim]

    # Subtract the projections from x1_batch
    x1_perp = x1_batch - projections.sum(dim=1)  # Shape: [batch_size, dim]

    # Normalize the resultant vectors
    x2_batch = x1_perp / torch.norm(x1_perp, dim=1, keepdim=True)  # Shape: [batch_size, dim]

    return x2_batch.to(basis_matrix.device)



class memory_base(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        n_cls = len(classnames)
        self.baseclss_feat =  nn.Parameter(torch.zeros(n_cls,512,dtype = clip_model.dtype))
        self.cfg = cfg
        
    def forward(self,x):
        self._momentum_update(x)
#         if self.cfg.DATASET.SUBSAMPLE_CLASSES!='base':
            
#         else:
#             baseclss_feat = self.baseclss_feat 
        
        # return baseclss_feat
    
    @torch.no_grad()
    def _momentum_update(self,x):
        self.baseclss_feat.data.copy_(x)
        

class CustomCLIP(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        self.prompt_learner = PromptLearner(cfg, classnames, clip_model)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype
        
        n_cls = len(classnames)
        self.add_class = int(0)
        self.n_cls = n_cls
        
        
        # self.fuse =self.prompt_learner.fuse
        # print(self.fuse)
        self.fuse = fuser(clip_model)
        
        # self.text_features_add = torch.randn(self.add_class,512).type(self.dtype)
        
        
        self.cfg =  cfg #novel/base if novel, we will callbirate it.
        self.Meom = memory_base(cfg, classnames, clip_model)
        
    def forward(self, image, label=None):
        tokenized_prompts = self.tokenized_prompts
        logit_scale = self.logit_scale.exp()

        image_features = self.image_encoder(image.type(self.dtype))
        
        
        # print('tokenized_prompts',tokenized_prompts.shape)
        prompts = self.prompt_learner()
        # print('prompts',prompts.shape)
        text_features = self.text_encoder(prompts, tokenized_prompts)
        
        # print('text_feautres:',text_feautres.shape)
        
        
        
        
        
        # _,image_features = self.fuse(text_features,image_features)
        
        
        
        # if self.training:
        #     self.text_features_add = torch.randn(self.add_class,512).type(self.dtype)
        #     text_features = torch.cat([text_features,self.text_features_add.to(text_features.device)])
        
        # if self.prompt_learner.training:
        #     text_features_center = text_features-text_features.mean(0,keepdims=True)#n_cls * d 
        #     text_features_center = text_features_center/ text_features_center.norm(dim=-1, keepdim=True)
        #     # div = torch.logdet(text_features_center.float()@text_features_center.T.float()+1e-8)
        #     div =  compute_nc2(text_features_center)
        
        
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        
        # image_features = (image_features-image_features.mean(1)) / image_features.std(1)
        # text_features = (text_features-text_features.mean(1)) / text_features.std(1)
        
        # if self.prompt_learner.training:
        #     self.Meom(text_features)
        # else:
        #     if self.cfg.DATASET.SUBSAMPLE_CLASSES!='base':
        #         baseclss_feat= self.Meom.baseclss_feat.to(text_features.device).type(text_features.dtype)
        #         print(baseclss_feat)
        #         text_features = find_orthogonal_min_sim_batch(text_features,baseclss_feat)
        
        # logits = logit_scale * image_features @ text_features.t()
        
        
        
        
        logits = logit_scale * image_features @ text_features.t()
        
        # logits = []
        # for pts_i, imf_i in zip(prompts, image_features):
        #     text_features = self.text_encoder(pts_i, tokenized_prompts)
        #     text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        #     l_i = logit_scale * imf_i @ text_features.t()
        #     logits.append(l_i)
        # logits = torch.stack(logits)
        
        
            
            
            
        if self.prompt_learner.training:
            # text_features = text_features-text_features.mean(0,keepdims=True)
            # _,s,_ = torch.linalg.svd(text_features.float())
            # div = torch.sum(torch.log(s[:self.n_cls]**2)).to(logits.device)
            # print("div:",div)
            
            return F.cross_entropy(logits, label) #+ 0*div.type(logits.dtype)

        return logits


@TRAINER_REGISTRY.register()
class CoCoOp(TrainerX):
    def check_cfg(self, cfg):
        assert cfg.TRAINER.COCOOP.PREC in ["fp16", "fp32", "amp"]

    def build_model(self):
        cfg = self.cfg
        classnames = self.dm.dataset.classnames

        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model = load_clip_to_cpu(cfg)

        if cfg.TRAINER.COCOOP.PREC == "fp32" or cfg.TRAINER.COCOOP.PREC == "amp":
            # CLIP's default precision is fp16
            clip_model.float()

        print("Building custom CLIP")
        self.model = CustomCLIP(cfg, classnames, clip_model)

        print("Turning off gradients in both the image and the text encoder")
        name_to_update = "prompt_learner"
        
        name_to_update2 = "fuse"
        
        for name, param in self.model.named_parameters():
            if name_to_update not in name and name_to_update2 not in name:
                param.requires_grad_(False)
        
        
        
        # Double check
        enabled = set()
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                enabled.add(name)
        print(f"Parameters to be updated: {enabled}")

        if cfg.MODEL.INIT_WEIGHTS:
            load_pretrained_weights(self.model.prompt_learner, cfg.MODEL.INIT_WEIGHTS)

        self.model.to(self.device)
        # NOTE: only give prompt_learner to the optimizer
        self.optim = build_optimizer(self.model.prompt_learner, cfg.OPTIM)
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
        self.register_model("prompt_learner", self.model.prompt_learner, self.optim, self.sched)
        self.register_model("fuse", self.model.fuse, self.optim, self.sched)
        self.register_model("memory_base", self.model, self.optim, self.sched)
        
        self.scaler = GradScaler() if cfg.TRAINER.COCOOP.PREC == "amp" else None

        # Note that multi-gpu training could be slow because CLIP's size is
        # big, which slows down the copy operation in DataParallel
        device_count = torch.cuda.device_count()
        if device_count > 1:
            print(f"Multiple GPUs detected (n_gpus={device_count}), use all of them!")
            self.model = nn.DataParallel(self.model)

    def forward_backward(self, batch):
        image, label = self.parse_batch_train(batch)

        model = self.model
        optim = self.optim
        scaler = self.scaler

        prec = self.cfg.TRAINER.COCOOP.PREC
        if prec == "amp":
            with autocast():
                loss = model(image, label)
            optim.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optim)
            scaler.update()
        else:
            loss = model(image, label)
            optim.zero_grad()
            loss.backward()
            optim.step()

        loss_summary = {"loss": loss.item()}

        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()

        return loss_summary

    def parse_batch_train(self, batch):
        input = batch["img"]
        label = batch["label"]
        
        # print(label)
        
        input = input.to(self.device)
        label = label.to(self.device)
        return input, label

    def load_model(self, directory, epoch=None):
        if not directory:
            print("Note that load_model() is skipped as no pretrained model is given")
            return

        names = self.get_model_names()

        # By default, the best model is loaded
        model_file = "model-best.pth.tar"

        if epoch is not None:
            model_file = "model.pth.tar-" + str(epoch)

        for name in names:
            model_path = osp.join(directory, name, model_file)

            if not osp.exists(model_path):
                raise FileNotFoundError('Model not found at "{}"'.format(model_path))

            checkpoint = load_checkpoint(model_path)
            state_dict = checkpoint["state_dict"]
            epoch = checkpoint["epoch"]

            # Ignore fixed token vectors
            if "token_prefix" in state_dict:
                del state_dict["token_prefix"]

            if "token_suffix" in state_dict:
                del state_dict["token_suffix"]

            print("Loading weights to {} " 'from "{}" (epoch = {})'.format(name, model_path, epoch))
            # set strict=False
            self._models[name].load_state_dict(state_dict, strict=False)
