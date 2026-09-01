#    Copyright 2023 Haotian Liu
#    Copyright 2026 Xiwen Chen
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.
#
#    Modifications: Added OTPrune visual token pruning via optimal transport.


from abc import ABC, abstractmethod

import torch
import torch.nn as nn

from .multimodal_encoder.builder import build_vision_tower
from .multimodal_projector.builder import build_vision_projector

from llava.constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_PATCH_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN

from llava.mm_utils import get_anyres_image_grid_shape

#divprune
import os 
class LlavaMetaModel:

    def __init__(self, config):
        super(LlavaMetaModel, self).__init__(config)

        if hasattr(config, "mm_vision_tower"):
            self.vision_tower = build_vision_tower(config, delay_load=True)
            self.mm_projector = build_vision_projector(config)

            if 'unpad' in getattr(config, 'mm_patch_merge_type', ''):
                self.image_newline = nn.Parameter(
                    torch.empty(config.hidden_size, dtype=self.dtype)
                )

    def get_vision_tower(self):
        vision_tower = getattr(self, 'vision_tower', None)
        if type(vision_tower) is list:
            vision_tower = vision_tower[0]
        return vision_tower

    def initialize_vision_modules(self, model_args, fsdp=None):
        vision_tower = model_args.vision_tower
        mm_vision_select_layer = model_args.mm_vision_select_layer
        mm_vision_select_feature = model_args.mm_vision_select_feature
        pretrain_mm_mlp_adapter = model_args.pretrain_mm_mlp_adapter
        mm_patch_merge_type = model_args.mm_patch_merge_type

        self.config.mm_vision_tower = vision_tower

        if self.get_vision_tower() is None:
            vision_tower = build_vision_tower(model_args)

            if fsdp is not None and len(fsdp) > 0:
                self.vision_tower = [vision_tower]
            else:
                self.vision_tower = vision_tower
        else:
            if fsdp is not None and len(fsdp) > 0:
                vision_tower = self.vision_tower[0]
            else:
                vision_tower = self.vision_tower
            vision_tower.load_model()

        self.config.use_mm_proj = True
        self.config.mm_projector_type = getattr(model_args, 'mm_projector_type', 'linear')
        self.config.mm_hidden_size = vision_tower.hidden_size
        self.config.mm_vision_select_layer = mm_vision_select_layer
        self.config.mm_vision_select_feature = mm_vision_select_feature
        self.config.mm_patch_merge_type = mm_patch_merge_type

        if getattr(self, 'mm_projector', None) is None:
            self.mm_projector = build_vision_projector(self.config)

            if 'unpad' in mm_patch_merge_type:
                embed_std = 1 / torch.sqrt(torch.tensor(self.config.hidden_size, dtype=self.dtype))
                self.image_newline = nn.Parameter(
                    torch.randn(self.config.hidden_size, dtype=self.dtype) * embed_std
                )
        else:
            # In case it is frozen by LoRA
            for p in self.mm_projector.parameters():
                p.requires_grad = True

        if pretrain_mm_mlp_adapter is not None:
            mm_projector_weights = torch.load(pretrain_mm_mlp_adapter, map_location='cpu')
            def get_w(weights, keyword):
                return {k.split(keyword + '.')[1]: v for k, v in weights.items() if keyword in k}

            self.mm_projector.load_state_dict(get_w(mm_projector_weights, 'mm_projector'))


def unpad_image(tensor, original_size):
    """
    Unpads a PyTorch tensor of a padded and resized image.

    Args:
    tensor (torch.Tensor): The image tensor, assumed to be in CxHxW format.
    original_size (tuple): The original size of PIL image (width, height).

    Returns:
    torch.Tensor: The unpadded image tensor.
    """
    original_width, original_height = original_size
    current_height, current_width = tensor.shape[1:]

    original_aspect_ratio = original_width / original_height
    current_aspect_ratio = current_width / current_height

    if original_aspect_ratio > current_aspect_ratio:
        scale_factor = current_width / original_width
        new_height = int(original_height * scale_factor)
        padding = (current_height - new_height) // 2
        unpadded_tensor = tensor[:, padding:current_height - padding, :]
    else:
        scale_factor = current_height / original_height
        new_width = int(original_width * scale_factor)
        padding = (current_width - new_width) // 2
        unpadded_tensor = tensor[:, :, padding:current_width - padding]

    return unpadded_tensor
def greedy_select(kernel_matrix, max_length, item_size, device, dtype, epsilon=1E-100):
    """
    Determinantal Point Process greedy selection.
    
    Args:
        kernel_matrix: (n x n) PSD kernel matrix
        max_length: maximum number of items to select
        item_size: size of the item pool (usually n)
        device: torch device
        dtype: torch dtype
        epsilon: small tolerance to stop selection
    
    Returns:
        selected_items: tensor of selected indices
    """
    cis  = torch.zeros((max_length, item_size), device=device, dtype=dtype)
    di2s = torch.clone(torch.diag(kernel_matrix))
    
    selected_items = []
    selected_item = torch.argmax(di2s).item()
    selected_items.append(selected_item)

    while len(selected_items) < max_length:
        k = len(selected_items) - 1
        ci_optimal = cis[:k, selected_item]
        di_optimal = torch.sqrt(di2s[selected_item])
        elements = kernel_matrix[selected_item, :]

        eis = (elements - (ci_optimal.unsqueeze(0) @ cis[:k, :]).squeeze()) / di_optimal
        cis[k, :] = eis
        di2s -= torch.square(eis)

        selected_item = torch.argmax(di2s).item()
        if di2s[selected_item] < epsilon:
            break
        selected_items.append(selected_item)

    return selected_items

class LlavaMetaForCausalLM(ABC):

    @abstractmethod
    def get_model(self):
        pass

    def get_vision_tower(self):
        return self.get_model().get_vision_tower()

    def encode_images(self, images):
        image_features = self.get_model().get_vision_tower()(images)
        image_features = self.get_model().mm_projector(image_features)
        return image_features

    # divprune
    def pairwise_cosine_similarity(self, matrix):
        norm_matrix = matrix / matrix.norm(dim=1, keepdim=True)
        cosine_similarity = torch.mm(norm_matrix, norm_matrix.t())
        return cosine_similarity

#     def DivPrune(self, visual_feature_vectors, image_feature_length, cosine_matrix=None, threshold_ratio=0.1):            
#         threshold_terms = int(round(threshold_ratio*image_feature_length))
#         if cosine_matrix is None:
#             cosine_matrix = 1.0 - (self.pairwise_cosine_similarity(visual_feature_vectors))

#         s = torch.empty(threshold_terms, dtype=torch.long, device=visual_feature_vectors.device)
#         for i in range(threshold_terms):
#             if i==0:
#                 m2 = cosine_matrix
#             else:
#                 m2 = torch.index_select(cosine_matrix, 0, torch.index_select(s,0,torch.arange(0,i,device=cosine_matrix.device)))

#             if i==0:
#                 scores = torch.topk(m2, 2,dim=0,largest=False).values[1,:] #for distance
#             else:
#                 scores = torch.min(m2, dim=0).values #for distance 

#             phrase_to_add_idx = torch.argmax(scores)
#             s[i] = phrase_to_add_idx
#         return s, cosine_matrix
    


    @torch.no_grad()
    def OTPrune(self, visual_feature_vectors, image_feature_length, cur_input_embeds_1, cosine_matrix=None, threshold_ratio=0.1, epsilon=1E-100): 
        gamma = 0.01
        ### cur_input_embeds_1  text embedding 
       
        device_cuda = visual_feature_vectors.device
        dtype = visual_feature_vectors.dtype
        
        
        cur_input_embeds_1 = cur_input_embeds_1 #.mean(dim=-2, keepdim=True) 
        # kernel_matrix =self.quad_form_with_rowspace_projector(visual_feature_vectors,cur_input_embeds_1)
        
        
        visual_feature_vectors = visual_feature_vectors / visual_feature_vectors.norm(dim=-2, keepdim=True).to(torch.float32)
        # gram = visual_feature_vectors.T @ visual_feature_vectors
        # U, S, Vh = torch.linalg.svd(gram)
        # sqrt_gram = (U * torch.sqrt(S)) @ Vh
        # visual_feature_vectors= visual_feature_vectors@sqrt_gram
        
        
        visual_feature_vectors= visual_feature_vectors@visual_feature_vectors.T #pre-computing
        # kernel_matrix = kernel_matrix+1e-9*torch.eyes(kernel_matrix.shape[0])
        m, d = visual_feature_vectors.shape
        
        
        
        
        
        max_length = int(round(threshold_ratio*image_feature_length))
        k = max_length
        gamma_tilde = gamma 
        
        
        kernel_matrix = torch.eye(image_feature_length).to(device_cuda)+ gamma_tilde*visual_feature_vectors@visual_feature_vectors.T
        
        item_size = image_feature_length
    

        selected_items = greedy_select(kernel_matrix, max_length, item_size, device_cuda, dtype, epsilon=1E-100)
        
        
        
        while len(set(selected_items))<max_length:
                
                for  i in selected_items:
                    kernel_matrix[i,:]=0
                    kernel_matrix[:,i]=0
                
                index_new = greedy_select(kernel_matrix, max_length-len(set(selected_items)), item_size, device_cuda, dtype, epsilon=1E-100)
                selected_items.extend(index_new)
        
        
        assert len(set(selected_items))==max_length
  
        selected_items = torch.tensor(list(set(selected_items)), dtype=torch.long, device=device_cuda)
        
        return selected_items, None
        
        
    
    
    def prepare_inputs_labels_for_multimodal(
        self, input_ids, position_ids, attention_mask, past_key_values, labels,
        images, image_sizes=None
    ):
        vision_tower = self.get_vision_tower()
        if vision_tower is None or images is None or input_ids.shape[1] == 1:
            return input_ids, position_ids, attention_mask, past_key_values, None, labels

        if type(images) is list or images.ndim == 5:
            if type(images) is list:
                images = [x.unsqueeze(0) if x.ndim == 3 else x for x in images]
            concat_images = torch.cat([image for image in images], dim=0)
            image_features = self.encode_images(concat_images)
            split_sizes = [image.shape[0] for image in images]
            image_features = torch.split(image_features, split_sizes, dim=0)
            mm_patch_merge_type = getattr(self.config, 'mm_patch_merge_type', 'flat')
            image_aspect_ratio = getattr(self.config, 'image_aspect_ratio', 'square')
            if mm_patch_merge_type == 'flat':
                image_features = [x.flatten(0, 1) for x in image_features]
            elif mm_patch_merge_type.startswith('spatial'):
                new_image_features = []
                for image_idx, image_feature in enumerate(image_features):
                    if image_feature.shape[0] > 1:
                        base_image_feature = image_feature[0]
                        image_feature = image_feature[1:]
                        height = width = self.get_vision_tower().num_patches_per_side
                        assert height * width == base_image_feature.shape[0]
                        if image_aspect_ratio == 'anyres':
                            num_patch_width, num_patch_height = get_anyres_image_grid_shape(image_sizes[image_idx], self.config.image_grid_pinpoints, self.get_vision_tower().config.image_size)
                            image_feature = image_feature.view(num_patch_height, num_patch_width, height, width, -1)
                        else:
                            raise NotImplementedError
                        if 'unpad' in mm_patch_merge_type:
                            image_feature = image_feature.permute(4, 0, 2, 1, 3).contiguous()
                            image_feature = image_feature.flatten(1, 2).flatten(2, 3)
                            image_feature = unpad_image(image_feature, image_sizes[image_idx])
                            image_feature = torch.cat((
                                image_feature,
                                self.model.image_newline[:, None, None].expand(*image_feature.shape[:-1], 1).to(image_feature.device)
                            ), dim=-1)
                            image_feature = image_feature.flatten(1, 2).transpose(0, 1)
                        else:
                            image_feature = image_feature.permute(0, 2, 1, 3, 4).contiguous()
                            image_feature = image_feature.flatten(0, 3)
                        image_feature = torch.cat((base_image_feature, image_feature), dim=0)
                    else:
                        image_feature = image_feature[0]
                        if 'unpad' in mm_patch_merge_type:
                            image_feature = torch.cat((
                                image_feature,
                                self.model.image_newline[None].to(image_feature.device)
                            ), dim=0)
                    new_image_features.append(image_feature)
                image_features = new_image_features
            else:
                raise ValueError(f"Unexpected mm_patch_merge_type: {self.config.mm_patch_merge_type}")
        else:
            image_features = self.encode_images(images)

        # TODO: image start / end is not implemented here to support pretraining.
        if getattr(self.config, 'tune_mm_mlp_adapter', False) and getattr(self.config, 'mm_use_im_start_end', False):
            raise NotImplementedError

        # Let's just add dummy tensors if they do not exist,
        # it is a headache to deal with None all the time.
        # But it is not ideal, and if you have a better idea,
        # please open an issue / submit a PR, thanks.
        _labels = labels
        _position_ids = position_ids
        _attention_mask = attention_mask
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
        else:
            attention_mask = attention_mask.bool()
        if position_ids is None:
            position_ids = torch.arange(0, input_ids.shape[1], dtype=torch.long, device=input_ids.device)
        if labels is None:
            labels = torch.full_like(input_ids, IGNORE_INDEX)

        # remove the padding using attention_mask -- FIXME
        _input_ids = input_ids
        input_ids = [cur_input_ids[cur_attention_mask] for cur_input_ids, cur_attention_mask in zip(input_ids, attention_mask)]
        labels = [cur_labels[cur_attention_mask] for cur_labels, cur_attention_mask in zip(labels, attention_mask)]

        new_input_embeds = []
        new_labels = []
        cur_image_idx = 0
        for batch_idx, cur_input_ids in enumerate(input_ids):
            num_images = (cur_input_ids == IMAGE_TOKEN_INDEX).sum()
            if num_images == 0:
                cur_image_features = image_features[cur_image_idx]
                cur_input_embeds_1 = self.get_model().embed_tokens(cur_input_ids)
                cur_input_embeds = torch.cat([cur_input_embeds_1, cur_image_features[0:0]], dim=0)
                new_input_embeds.append(cur_input_embeds)
                new_labels.append(labels[batch_idx])
                cur_image_idx += 1
                continue

            image_token_indices = [-1] + torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0].tolist() + [cur_input_ids.shape[0]]
            cur_input_ids_noim = []
            cur_labels = labels[batch_idx]
            cur_labels_noim = []
            for i in range(len(image_token_indices) - 1):
                cur_input_ids_noim.append(cur_input_ids[image_token_indices[i]+1:image_token_indices[i+1]])
                cur_labels_noim.append(cur_labels[image_token_indices[i]+1:image_token_indices[i+1]])
            split_sizes = [x.shape[0] for x in cur_labels_noim]
            cur_input_embeds = self.get_model().embed_tokens(torch.cat(cur_input_ids_noim))
            cur_input_embeds_no_im = torch.split(cur_input_embeds, split_sizes, dim=0)
            cur_new_input_embeds = []
            cur_new_labels = []

            for i in range(num_images + 1):
                cur_new_input_embeds.append(cur_input_embeds_no_im[i])
                cur_new_labels.append(cur_labels_noim[i])
                if i < num_images:
                    cur_image_features = image_features[cur_image_idx]
                    cur_image_idx += 1
                    cur_new_input_embeds.append(cur_image_features)
                    cur_new_labels.append(torch.full((cur_image_features.shape[0],), IGNORE_INDEX, device=cur_labels.device, dtype=cur_labels.dtype))

            cur_new_input_embeds = [x.to(self.device) for x in cur_new_input_embeds]

            cur_new_input_embeds = torch.cat(cur_new_input_embeds)
            cur_new_labels = torch.cat(cur_new_labels)

            new_input_embeds.append(cur_new_input_embeds)
            new_labels.append(cur_new_labels)

        # Truncate sequences to max length as image embeddings can make the sequence longer
        tokenizer_model_max_length = getattr(self.config, 'tokenizer_model_max_length', None)
        if tokenizer_model_max_length is not None:
            new_input_embeds = [x[:tokenizer_model_max_length] for x in new_input_embeds]
            new_labels = [x[:tokenizer_model_max_length] for x in new_labels]

        # Combine them
        max_len = max(x.shape[0] for x in new_input_embeds)
        batch_size = len(new_input_embeds)

        new_input_embeds_padded = []
        new_labels_padded = torch.full((batch_size, max_len), IGNORE_INDEX, dtype=new_labels[0].dtype, device=new_labels[0].device)
        attention_mask = torch.zeros((batch_size, max_len), dtype=attention_mask.dtype, device=attention_mask.device)
        position_ids = torch.zeros((batch_size, max_len), dtype=position_ids.dtype, device=position_ids.device)

        for i, (cur_new_embed, cur_new_labels) in enumerate(zip(new_input_embeds, new_labels)):
            cur_len = cur_new_embed.shape[0]
            if getattr(self.config, 'tokenizer_padding_side', 'right') == "left":
                new_input_embeds_padded.append(torch.cat((
                    torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device),
                    cur_new_embed
                ), dim=0))
                if cur_len > 0:
                    new_labels_padded[i, -cur_len:] = cur_new_labels
                    attention_mask[i, -cur_len:] = True
                    position_ids[i, -cur_len:] = torch.arange(0, cur_len, dtype=position_ids.dtype, device=position_ids.device)
            else:
                new_input_embeds_padded.append(torch.cat((
                    cur_new_embed,
                    torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device)
                ), dim=0))
                if cur_len > 0:
                    new_labels_padded[i, :cur_len] = cur_new_labels
                    attention_mask[i, :cur_len] = True
                    position_ids[i, :cur_len] = torch.arange(0, cur_len, dtype=position_ids.dtype, device=position_ids.device)

        new_input_embeds = torch.stack(new_input_embeds_padded, dim=0)

        if _labels is None:
            new_labels = None
        else:
            new_labels = new_labels_padded

        if _attention_mask is None:
            attention_mask = None
        else:
            attention_mask = attention_mask.to(dtype=_attention_mask.dtype)

        if _position_ids is None:
            position_ids = None

        #divprune
        if 'LAYER_INDEX' in os.environ:
            #print("I am called without layer 0")
            if type(image_features) == list: #this is for LLaVA 1.6
                img_feature_len = image_features[0].shape[0] #example is 2340x4096
            else: #for LLaVa 1.5
                img_feature_len = image_features.shape[1] 

            if hasattr(self.config, 'img_feature_len'):
                self.config.img_feature_len = img_feature_len
            else:
                setattr(self.config, 'img_feature_len', img_feature_len)

        if 'LAYER_INDEX' in os.environ and os.environ['LAYER_INDEX']=='0':
            SYS_TOKEN_LEN = 35 
            diverse_ratio = float(os.environ['SUBSET_RATIO']) #define the subset selection ratio
            cosine_matrix = None
            if type(image_features) == list: #this is for LLaVA 1.6
                img_feature_len = image_features[0].shape[0] #example is 2340x4096
            else: #for LLaVa 1.5
                img_feature_len = image_features.shape[1] #example is 2340x4096

#             visual_tokens =new_input_embeds[0][SYS_TOKEN_LEN:SYS_TOKEN_LEN+img_feature_len]
#             selected_visual_tokens, cosine_matrix = self.DivPrune(visual_tokens, img_feature_len,cosine_matrix,threshold_ratio=diverse_ratio)
#             
                      
#             selected_visual_tokens += SYS_TOKEN_LEN
#             keep_indexs = torch.cat((torch.arange(SYS_TOKEN_LEN,device=new_input_embeds.device), selected_visual_tokens, torch.arange(SYS_TOKEN_LEN+img_feature_len,new_input_embeds.shape[1],device=new_input_embeds.device)))
#             keep_indexs = keep_indexs.sort().values
            
            visual_tokens =new_input_embeds[0][SYS_TOKEN_LEN:SYS_TOKEN_LEN+img_feature_len]
            text_tokens = new_input_embeds[0][SYS_TOKEN_LEN+img_feature_len:]
            selected_visual_tokens,_  = self.OTPrune(visual_tokens,img_feature_len,text_tokens,cosine_matrix,threshold_ratio=diverse_ratio)

            
            # assert img_feature_len==len(selected_visual_tokens) ,img_feature_len
            # assert int(round(threshold_ratio*image_feature_length))==len(selected_visual_tokens) ,diverse_ratio
            
            
            selected_visual_tokens += SYS_TOKEN_LEN
            keep_indexs = torch.cat((torch.arange(SYS_TOKEN_LEN,device=new_input_embeds.device), selected_visual_tokens, torch.arange(SYS_TOKEN_LEN+img_feature_len,new_input_embeds.shape[1],device=new_input_embeds.device)))
    
            new_input_embeds = new_input_embeds[:,keep_indexs]
            if position_ids is not None:
                position_ids = position_ids[:,keep_indexs,:]
            if attention_mask is not None:
                attention_mask = attention_mask[:,keep_indexs]
        return None, position_ids, attention_mask, past_key_values, new_input_embeds, new_labels

    def initialize_vision_tokenizer(self, model_args, tokenizer):
        if model_args.mm_use_im_patch_token:
            tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
            self.resize_token_embeddings(len(tokenizer))

        if model_args.mm_use_im_start_end:
            num_new_tokens = tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)
            self.resize_token_embeddings(len(tokenizer))

            if num_new_tokens > 0:
                input_embeddings = self.get_input_embeddings().weight.data
                output_embeddings = self.get_output_embeddings().weight.data

                input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(
                    dim=0, keepdim=True)
                output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
                    dim=0, keepdim=True)

                input_embeddings[-num_new_tokens:] = input_embeddings_avg
                output_embeddings[-num_new_tokens:] = output_embeddings_avg

            if model_args.tune_mm_mlp_adapter:
                for p in self.get_input_embeddings().parameters():
                    p.requires_grad = True
                for p in self.get_output_embeddings().parameters():
                    p.requires_grad = False

            if model_args.pretrain_mm_mlp_adapter:
                mm_projector_weights = torch.load(model_args.pretrain_mm_mlp_adapter, map_location='cpu')
                embed_tokens_weight = mm_projector_weights['model.embed_tokens.weight']
                assert num_new_tokens == 2
                if input_embeddings.shape == embed_tokens_weight.shape:
                    input_embeddings[-num_new_tokens:] = embed_tokens_weight[-num_new_tokens:]
                elif embed_tokens_weight.shape[0] == num_new_tokens:
                    input_embeddings[-num_new_tokens:] = embed_tokens_weight
                else:
                    raise ValueError(f"Unexpected embed_tokens_weight shape. Pretrained: {embed_tokens_weight.shape}. Current: {input_embeddings.shape}. Numer of new tokens: {num_new_tokens}.")
        elif model_args.mm_use_im_patch_token:
            if model_args.tune_mm_mlp_adapter:
                for p in self.get_input_embeddings().parameters():
                    p.requires_grad = False
                for p in self.get_output_embeddings().parameters():
                    p.requires_grad = False
