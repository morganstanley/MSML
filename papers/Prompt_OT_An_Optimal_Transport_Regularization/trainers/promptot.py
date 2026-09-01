import copy
import os.path as osp
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.cuda.amp import GradScaler, autocast

from dassl.engine import TRAINER_REGISTRY, TrainerX
from dassl.utils import load_pretrained_weights, load_checkpoint
from dassl.optim import build_optimizer, build_lr_scheduler
from clip import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
from .imagenet_templates import IMAGENET_TEMPLATES

from models.linearatt import MultiheadLinearAttention
from torch.nn import MultiheadAttention
# from timm.optim import Lookahead

_tokenizer = _Tokenizer()


def load_clip_to_cpu(cfg, zero_shot_model=False):
    backbone_name = cfg.MODEL.BACKBONE.NAME
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url)

    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")
    if not zero_shot_model:
        design_details = {"trainer": 'IVLP',
                          "vision_depth": cfg.TRAINER.PROMPTOT.PROMPT_DEPTH_VISION,
                          "language_depth": cfg.TRAINER.PROMPTOT.PROMPT_DEPTH_TEXT,
                          "vision_ctx": cfg.TRAINER.PROMPTOT.N_CTX_VISION,
                          "language_ctx": cfg.TRAINER.PROMPTOT.N_CTX_TEXT}
        model = clip.build_model(state_dict or model.state_dict(), design_details)
    else:
        # Return original CLIP model for generating frozen VL features
        design_details = {"trainer": 'IVLP',
                          "vision_depth": 0,
                          "language_depth": 0, "vision_ctx": 0,
                          "language_ctx": 0}
        model = clip.build_model(state_dict or model.state_dict(), design_details)
        return model
    return model


class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
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


class VLPromptLearner(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        n_cls = len(classnames)
        # Make sure Language depth >= 1
        assert cfg.TRAINER.PROMPTOT.PROMPT_DEPTH_TEXT >= 1, "In Independent VL prompting, Language prompt depth should be >=1" \
                                                        "\nPlease use VPT trainer if you want to learn only vision " \
                                                        "branch"
        n_ctx = cfg.TRAINER.PROMPTOT.N_CTX_TEXT
        ctx_init = cfg.TRAINER.PROMPTOT.CTX_INIT
        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]
        clip_imsize = clip_model.visual.input_resolution
        cfg_imsize = cfg.INPUT.SIZE[0]
        assert cfg_imsize == clip_imsize, f"cfg_imsize ({cfg_imsize}) must equal to clip_imsize ({clip_imsize})"

        if ctx_init and n_ctx <= 4:
            # use given words to initialize context vectors
            ctx_init = ctx_init.replace("_", " ")
            n_ctx = n_ctx
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
        print(f"Independent V-L design")
        print(f'Initial text context: "{prompt_prefix}"')
        print(f"Number of context words (tokens) for Language prompting: {n_ctx}")
        print(f"Number of context words (tokens) for Vision prompting: {cfg.TRAINER.PROMPTOT.N_CTX_VISION}")
        self.ctx = nn.Parameter(ctx_vectors)

        classnames = [name.replace("_", " ") for name in classnames]
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        prompts = [prompt_prefix + " " + name + "." for name in classnames]

        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])  # (n_cls, n_tkn)
        # Also create frozen CLIP
        clip_model_temp = load_clip_to_cpu(cfg, True).float().cuda()
        clip_model_temp_image = load_clip_to_cpu(cfg, True)
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)
            self.ZS_image_encoder = clip_model_temp_image.visual
            # Now pre-compute the frozen VL embeddings
            all_teacher_features = []
            # Using multiple text templates to ensure textual diversity during training
            for single_template in IMAGENET_TEMPLATES:
                x = [single_template.replace("{}", name) for name in classnames]
                x_tokenized = torch.cat([clip.tokenize(p) for p in x])
                text_features = clip_model_temp.encode_text(x_tokenized.cuda())
                all_teacher_features.append(text_features.unsqueeze(1))
        # print(all_teacher_features,torch.cat(all_teacher_features, dim=1).shape)
        self.fixed_embeddings = torch.cat(all_teacher_features, dim=1)#.mean(dim=1)
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

    def forward(self):
        ctx = self.ctx
        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)

        prefix = self.token_prefix
        suffix = self.token_suffix
        prompts = self.construct_prompts(ctx, prefix, suffix)

        return prompts
    
    
import ot

class miniBatchOT(nn.Module):
    def __init__(self, tau=0.1,epsilon=0,iteration=10,eta1=0.5,eta2=0.5,methods="jdot"):
        super(miniBatchOT, self).__init__()
        self.tau = tau
        self.epsilon = epsilon
        self.iteration = iteration
        self.methods = methods
        self.eta1 = eta1
        self.eta2 = eta2

    def forward(self, feature, zero_feature,text,zero_text,groundtruth,logits):
        feature, zero_feature,logits = feature.float(), zero_feature.float(),logits.float()
        
        #joint representation of adapated model
        feature = torch.cat([feature,text[groundtruth]],dim=1)
        
        
        zero_feature = zero_feature.squeeze(1)
        #joint representation of zero-shot model
        zero_feature = torch.cat([zero_feature,zero_text[groundtruth]],dim=1)
        
        
       
        embed_cost = torch.cdist(feature, zero_feature) ** 2
        
        
        
        
        
        
      
        
        total_cost = self.eta1 * embed_cost 
        total_cost = total_cost.cuda()
        
        
        
        a, b = ot.unif(feature.size()[0]), ot.unif(zero_feature.size()[0])
      
        if self.methods == "jumbot":
            pi = ot.unbalanced.sinkhorn_knopp_unbalanced(a, b, total_cost.detach().cpu().numpy(), self.epsilon, self.tau)
        elif self.methods == "jdot":
            if self.epsilon == 0:
                pi = ot.emd(a, b, total_cost.detach().cpu().numpy())
            else:
                pi = ot.sinkhorn(a, b, total_cost.detach().cpu().numpy(), reg=self.epsilon)
        elif self.methods == "jpmbot":
            if self.epsilon == 0:
                pi = ot.partial.partial_wasserstein(a, b, total_cost.detach().cpu().numpy(), self.mass)
            else:
                pi = ot.partial.entropic_partial_wasserstein(a, b, total_cost.detach().cpu().numpy(), m=self.mass, reg=self.epsilon)
                
        pi = torch.from_numpy(pi).float().cuda()
        #print(pi)
        ot_loss = torch.sum(pi * total_cost)   
             
        return ot_loss    

    
    
class CustomCLIP(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        self.prompt_learner = VLPromptLearner(cfg, classnames, clip_model)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype
        self.total_epochs = cfg.OPTIM.MAX_EPOCH
        self.n_cls = len(classnames)

        
        
        self.div = cfg.div
        
        self.cfg = cfg
    
    def forward(self, image, label=None):
        tokenized_prompts = self.tokenized_prompts
        logit_scale = self.logit_scale.exp()

        prompts = self.prompt_learner()
        # Compute the prompted image and text features
        text_features = self.text_encoder(prompts, tokenized_prompts)
        image_features = self.image_encoder(image.type(self.dtype))
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        # Compute the prompted logits
        # logits = logit_scale * image_features @ text_features.t()
        # if self.prompt_learner.training:
            # Now calculate the frozen pre-trained features
        fixed_embeddings = self.prompt_learner.fixed_embeddings  # [CLASS, templet, DIM] precomputed pre-trained frozen textual features
        fixed_embeddings = fixed_embeddings / fixed_embeddings.norm(dim=-1, keepdim=True)
        fixed_embeddings = fixed_embeddings.half()
        
      
            
        if self.prompt_learner.training:   
            with torch.no_grad():
                zero_shot_features = self.prompt_learner.ZS_image_encoder(image.type(self.dtype))
                zero_shot_features = zero_shot_features / zero_shot_features.norm(dim=-1, keepdim=True).half()
                # Compute pre-trained frozen visual features
                # zero_shot_logits = logit_scale * zero_shot_features.cuda() @ fixed_embeddings.half().cuda().t()
                
                #zero_shot_features B*512
                #fixed_embeddings B*60*612
                

            
            zero_shot_features = zero_shot_features.unsqueeze(1)
            zero_shot_features0 =zero_shot_features
            fixed_embeddings0 =fixed_embeddings


            zero_shot_logits =logit_scale*torch.matmul(zero_shot_features0.unsqueeze(2), fixed_embeddings0.unsqueeze(0).transpose(-1, -2)).squeeze(2).permute(0,2,1)
            # print("zero_shot_logits:",zero_shot_logits.shape) #4*60*51
            
        

            logits = logit_scale * image_features @ text_features.t()

            
        
   
                
                
            
        else:
            
            logits = logit_scale * image_features @ text_features.t()
        
        if self.prompt_learner.training:    
            return F.cross_entropy(logits,
                                   label), text_features, fixed_embeddings[:,0,:], zero_shot_features0, \
                   image_features, zero_shot_logits, logits
        else:

            
            
            return logits


                
@TRAINER_REGISTRY.register()
class PromptOT(TrainerX):
    def check_cfg(self, cfg):
        assert cfg.TRAINER.PROMPTOT.PREC in ["fp16", "fp32", "amp"]

    def build_model(self):
        cfg = self.cfg
        classnames = self.dm.dataset.classnames
        
        self.div = cfg.div  ###lamda = 10
        
        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model = load_clip_to_cpu(cfg)

        if cfg.TRAINER.PROMPTOT.PREC == "fp32" or cfg.TRAINER.PROMPTOT.PREC == "amp":
            # CLIP's default precision is fp16
            clip_model.float()

        print("Building custom CLIP")
        self.model = CustomCLIP(cfg, classnames, clip_model)

        print("Turning off gradients in both the image and the text encoder")
        name_to_update = "prompt_learner"
        name_to_update2 = "regularizer"
        name_to_update3 = "logit_regularizer"
        
        for name, param in self.model.named_parameters():
            if name_to_update not in name: #logit_regularizer
                # Make sure that VPT prompts are updated
                if "VPT" in name:
                    param.requires_grad_(True)
                else:
                    param.requires_grad_(False)
            else:
                if "ZS_image_encoder" in name:
                    param.requires_grad_(False)

        # Double check
        enabled = set()
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                enabled.add(name)
        print(f"Parameters to be updated: {enabled}")
        print(f"Parameters count: {len(enabled)}")
        if cfg.MODEL.INIT_WEIGHTS:
            load_pretrained_weights(self.model, cfg.MODEL.INIT_WEIGHTS)

        self.model.to(self.device)
        # NOTE: only give prompt_learner to the optimizer
        self.optim = build_optimizer(self.model, cfg.OPTIM)
        # if cfg.OPTIM != 'sgd':
        #     self.optim = Lookahead(self.optim,alpha=0.1,k=8)
        
        
        
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
        self.register_model("VLPromptLearner", self.model, self.optim, self.sched)
       
        
        # Cosine scheduler
        self.total_epochs = cfg.OPTIM.MAX_EPOCH
        self.step_counter = 1
        N = cfg.OPTIM.MAX_EPOCH
        # mean = cfg.TRAINER.PROMPTOT.GPA_MEAN
        # stdev = cfg.TRAINER.PROMPTOT.GPA_STD
        # gauss = self.get_gauss(mean, stdev)
        # self.gauss = np.array([gauss(a) for a in range(1, N + 1)])
        # self.gauss = self.gauss / sum(self.gauss)
        self.scaler = GradScaler() if cfg.TRAINER.PROMPTOT.PREC == "amp" else None
        # Note that multi-gpu training could be slow because CLIP's size is
        # big, which slows down the copy operation in DataParallel
        device_count = torch.cuda.device_count()
        if device_count > 1:
            print(f"Multiple GPUs detected (n_gpus={device_count}), use all of them!")
            self.model = nn.DataParallel(self.model)
        
        self.previous_model_gpa = None

    def forward_backward(self, batch):
        image, label = self.parse_batch_train(batch)

        model = self.model
        optim = self.optim
        scaler = self.scaler

        prec = self.cfg.TRAINER.PROMPTOT.PREC
        if prec == "amp":
            with autocast():
                loss = model(image, label)
            optim.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optim)
            scaler.update()
        else:
            loss_ce, normalized_text_features, zs_clip_text_embeddings, zs_image_embedd, image_ft, \
            zero_shot_logits, logits = model(image, label)
            
            loss1 = miniBatchOT()(image_ft, zs_image_embedd,normalized_text_features,zs_clip_text_embeddings,label,logits)#jot loss
           
            

            
            
            loss = loss_ce+self.div*loss1#+self.div*loss2  #loss_ce+(loss_ce + L_SCL)
            optim.zero_grad()
            loss.backward()
            optim.step()

        loss_summary = {"loss": loss.item()}

        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()
            # Means one epoch is completed, perform GPA
            self.step_counter = self.step_counter + 1
            # current_epoch_weight = self.gauss[self.step_counter - 2]
            # current_model_weights = copy.deepcopy(model.state_dict())
            # weighted_state_dict = self.state_dict_weighting(current_model_weights, current_epoch_weight)
            # if self.previous_model_gpa is None:
            #     self.previous_model_gpa = weighted_state_dict
            # else:
            #     self.previous_model_gpa = self.state_dict_add(weighted_state_dict, self.previous_model_gpa)

        
            
            
        return loss_summary

    

   

    def parse_batch_train(self, batch):
        input = batch["img"]
        label = batch["label"]
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
            if "prompt_learner.token_prefix" in state_dict:
                del state_dict["prompt_learner.token_prefix"]

            if "prompt_learner.token_suffix" in state_dict:
                del state_dict["prompt_learner.token_suffix"]

            print("Loading weights to {} " 'from "{}" (epoch = {})'.format(name, model_path, epoch))
            # set strict=False
            self._models[name].load_state_dict(state_dict, strict=False)