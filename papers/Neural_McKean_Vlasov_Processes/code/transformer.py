
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

def get_emb(sin_inp):
    """
    Gets a base embedding for one dimension with sin and cos intertwined
    """
    emb = torch.stack((sin_inp.sin(), sin_inp.cos()), dim=-1)
    return torch.flatten(emb, -2, -1)


class PositionalEncoding(nn.Module):
    def __init__(self, channels):
        """
        :param channels: The last dimension of the tensor you want to apply pos emb to.
        """
        super(PositionalEncoding, self).__init__()
        self.org_channels = channels
        channels = int(math.ceil(channels / 2) * 2)
        self.channels = channels
        inv_freq = 1.0 / (10000 ** (torch.arange(0, channels, 2).float() / channels))
        self.register_buffer("inv_freq", inv_freq)
        self.cached_penc = None

    def forward(self, tensor):
        """
        :param tensor: A 3d tensor of size (batch_size, x, ch)
        :return: Positional Encoding Matrix of size (batch_size, x, ch)
        """
        if len(tensor.shape) != 3:
            raise RuntimeError("The input tensor has to be 3d!")

        if self.cached_penc is not None and self.cached_penc.shape == tensor.shape:
            return self.cached_penc

        self.cached_penc = None
        batch_size, x, orig_ch = tensor.shape
        pos_x = torch.arange(x, device=tensor.device).type(self.inv_freq.type())
        sin_inp_x = torch.einsum("i,j->ij", pos_x, self.inv_freq)
        emb_x = get_emb(sin_inp_x)
        emb = torch.zeros((x, self.channels), device=tensor.device).type(tensor.type())
        emb[:, : self.channels] = emb_x

        self.cached_penc = emb[None, :, :orig_ch].repeat(batch_size, 1, 1)
        return self.cached_penc

class TransformerModel(nn.Module):
    """Container module with an encoder, a recurrent or transformer module, and a decoder."""

    def __init__(self, ninp, nlayers, nhid, dim_val=16, nhead=4, dropout=0.5):
        super(TransformerModel, self).__init__()
        try:
            from torch.nn import TransformerEncoder, TransformerEncoderLayer
        except BaseException as e:
            raise ImportError('TransformerEncoder module does not exist in PyTorch 1.1 or '
                              'lower.') from e
        self.model_type = 'Transformer'
        self.src_mask = None
        #self.pos_encoder = PositionalEncoding(ninp)
        self.transformer_encoder = TimeSeriesTransformer(ninp, nlayers, nhid, dim_val=dim_val, nhead=nhead, batch_first = True, dropout = dropout)
        self.ninp = ninp
        
        self.decoder = nn.Linear(dim_val, ninp)
        self.decoder_var = nn.Linear(dim_val, ninp)

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, src, t=None, hidden=None, src_mask=None, tgt_mask = None):
        #src = self.pos_encoder(src)
        
        device = src.device
        mask = self._generate_square_subsequent_mask(len(src[:,-2:,:])).to(device)
        self.tgt_mask = mask
        
        output_temp = self.transformer_encoder(src[:,:-1,:], src[:,-2:,:], None, tgt_mask)
        output = self.decoder(output_temp)
        output_var = F.softplus(self.decoder_var(output_temp))
        if self.training:
            return output, output_var, None
        else:
            return output[:, -1, :], output_var[:, -1, :], None

from torch import Tensor
class TimeSeriesTransformer(nn.Module):
    def __init__(self, 
        ninp: int,
        nlayers: int,
        nhid: int, 
        dim_val: int,
        nhead: int,
        dec_seq_len: int=10,
        batch_first: bool=True,
        n_encoder_layers: int=4,
        n_decoder_layers: int=4,
        dropout_encoder: float=0.2, 
        dropout_decoder: float=0.2,
        dropout_pos_enc: float=0.1,
        dim_feedforward_encoder: int=2048,
        dim_feedforward_decoder: int=2048,
        num_predicted_features: int=1,
        **params
        ): 


        super().__init__() 
        num_predicted_features = ninp
        n_encoder_layers = n_decoder_layers = nlayers
        dim_feedforward_encoder = dim_feedforward_decoder = nhid
        self.dec_seq_len = dec_seq_len

        # Creating the three linear layers needed for the model

        self.encoder_input_layer = nn.Linear(in_features=ninp, out_features=dim_val)
        encoder_layer = nn.TransformerEncoderLayer(d_model=dim_val, nhead=nhead,dim_feedforward=dim_feedforward_encoder,dropout=dropout_encoder,batch_first=batch_first)
        self.encoder = nn.TransformerEncoder(encoder_layer=encoder_layer,num_layers=n_encoder_layers, norm=None)

        self.decoder_input_layer = nn.Linear(in_features=num_predicted_features,out_features=dim_val)  
        decoder_layer = nn.TransformerDecoderLayer(d_model=dim_val,nhead=nhead,dim_feedforward=dim_feedforward_decoder,dropout=dropout_decoder,batch_first=batch_first)
        self.decoder = nn.TransformerDecoder(decoder_layer=decoder_layer,num_layers=n_decoder_layers, norm=None)

    def forward(self, src: Tensor, tgt: Tensor, src_mask: Tensor=None, 
                tgt_mask: Tensor=None) -> Tensor:
        src = self.encoder_input_layer(src)
        src = self.encoder(src=src)
        
        decoder_output = self.decoder_input_layer(tgt)
        decoder_output = self.decoder(tgt=decoder_output,memory=src,tgt_mask=tgt_mask,memory_mask=src_mask)

        return decoder_output