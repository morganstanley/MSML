
import torch
from models.utils import *

from ipdb import set_trace as debug

def build_unetv2(config, zero_out_last_layer):
    # attention_resolutions = config.attention_resolutions
    attention_layers = config.attention_layers
    in_channels=config.in_channels
    out_channel=config.out_channel
    num_head=config.num_head
    num_res_blocks=config.num_res_blocks
    num_channels=config.num_channels
    dropout=config.dropout
    num_norm_groups=config.num_norm_groups
    channel_mult = config.channel_mult
    # image_size = config.image_size
    input_size = config.input_size


    # attention_ds = []
    # for res in attention_resolutions.split(","):
    #     attention_ds.append(image_size // int(res))
    # print('attention_ds', attention_ds)

    return UNetModelv2(
        input_size=input_size,
        in_channels=in_channels,
        model_channels=num_channels,
        out_channels=out_channel,
        num_res_blocks=num_res_blocks,
        # attention_resolutions=tuple(attention_ds),
        attention_layers=attention_layers,
        dropout=dropout,
        channel_mult=channel_mult,
        num_classes=None,
        use_checkpoint=False,
        num_heads=num_head,
        num_heads_upsample=-1,
        num_norm_groups=num_norm_groups,
        use_scale_shift_norm=True,
        zero_out_last_layer=zero_out_last_layer,
    )

class UNetModelv2(nn.Module):
    """
    The full UNet model with attention and timestep embedding.

    :param in_channels: channels in the input Tensor.
    :param model_channels: base channel count for the model.
    :param out_channels: channels in the output Tensor.
    :param num_res_blocks: number of residual blocks per downsample.
    :param attention_resolutions: a collection of downsample rates at which
        attention will take place. May be a set, list, or tuple.
        For example, if this contains 4, then at 4x downsampling, attention
        will be used.
    :param dropout: the dropout probability.
    :param channel_mult: channel multiplier for each level of the UNet.
    :param conv_resample: if True, use learned convolutions for upsampling and
        downsampling.
    :param dims: determines if the signal is 1D, 2D, or 3D.
    :param num_classes: if specified (as an int), then this model will be
        class-conditional with `num_classes` classes.
    :param use_checkpoint: use gradient checkpointing to reduce memory usage.
    :param num_heads: the number of attention heads in each attention layer.
    """

    def __init__(
        self,
        input_size,
        in_channels,
        model_channels,
        out_channels,
        num_res_blocks,
        # attention_resolutions,
        attention_layers,
        dropout=0,
        channel_mult=(1, 2, 4, 8),
        conv_resample=True,
        dims=2,
        num_classes=None,
        use_checkpoint=False,
        num_heads=1,
        num_heads_upsample=-1,
        num_norm_groups=32,
        use_scale_shift_norm=False,
        zero_out_last_layer=False,
    ):
        super().__init__()
        if num_heads_upsample == -1:
            num_heads_upsample = num_heads
        self.input_size = input_size
        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks
        # self.attention_resolutions = attention_resolutions
        self.attention_layers = attention_layers
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.num_scales = len(channel_mult)
        self.conv_resample = conv_resample
        self.num_classes = num_classes
        self.use_checkpoint = use_checkpoint
        self.num_heads = num_heads
        self.num_heads_upsample = num_heads_upsample
        self.zero_out_last_layer = zero_out_last_layer
        # self.layer_sizes = layer_sizes = [
        #     (input_size[0] // (2 ** i), input_size[1] // (2 ** i))
        #     for i in range(self.num_scales)]
        self.layer_sizes = self.input_sizes_on_scales(input_size, self.num_scales)

        print('layer_sizes: ', self.layer_sizes)

        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            nn.Linear(model_channels, time_embed_dim),
            SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )

        if self.num_classes is not None:
            self.label_emb = nn.Embedding(num_classes, time_embed_dim)

        self.input_blocks = nn.ModuleList(
            [
                TimestepEmbedSequential(
                    conv_nd(dims, in_channels, model_channels, 3, padding=1)
                )
            ]
        )
        input_block_chans = [model_channels]
        ch = model_channels
        ds = 1
        # print('-------- Downsample block ---------')
        for level, mult in enumerate(channel_mult):
            for _ in range(num_res_blocks):
                layers = [
                    ResBlock(
                        ch,
                        time_embed_dim,
                        dropout,
                        out_channels=mult * model_channels,
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                        num_norm_groups=num_norm_groups,
                    )
                ]
                ch = mult * model_channels
                # print('level', level, 'mult', mult, 'ds', ds)
                # if ds in attention_resolutions:
                if level in attention_layers:
                    # print('add attn')
                    layers.append(
                        AttentionBlock(
                            ch,
                            use_checkpoint=use_checkpoint,
                            num_heads=num_heads,
                            num_norm_groups=num_norm_groups,
                        )
                    )
                self.input_blocks.append(TimestepEmbedSequential(*layers))
                input_block_chans.append(ch)
            if level != len(channel_mult) - 1:
                self.input_blocks.append(
                    TimestepEmbedSequential(Downsample(ch, conv_resample, dims=dims))
                )
                input_block_chans.append(ch)
                ds *= 2

        # print('-------- middle line ---------')
        self.middle_block = TimestepEmbedSequential(
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
                num_norm_groups=num_norm_groups,
            ),
            AttentionBlock(
                ch,
                use_checkpoint=use_checkpoint,
                num_heads=num_heads,
                num_norm_groups=num_norm_groups,
            ),
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
                num_norm_groups=num_norm_groups,
            ),
        )

        # print('-------- Upsample block ---------')
        self.output_blocks = nn.ModuleList([])
        for level, mult in list(enumerate(channel_mult))[::-1]:
            for i in range(num_res_blocks + 1):
                layers = [
                    ResBlock(
                        ch + input_block_chans.pop(),
                        time_embed_dim,
                        dropout,
                        out_channels=model_channels * mult,
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                        num_norm_groups=num_norm_groups,
                    )
                ]
                ch = model_channels * mult
                # print('level', level, 'mult', mult, 'ds', ds)
                # if ds in attention_resolutions:
                if level in attention_layers:
                    layers.append(
                        AttentionBlock(
                            ch,
                            use_checkpoint=use_checkpoint,
                            num_heads=num_heads_upsample,
                            num_norm_groups=num_norm_groups,
                        )
                    )
                if level and i == num_res_blocks:
                    # layers.append(Upsample(ch, conv_resample, dims=dims))
                    layers.append(Upsamplev2(ch, conv_resample, dims=dims,
                        output_size=self.layer_sizes[level-1]))  # level-1, output==next level size.
                    ds //= 2
                self.output_blocks.append(TimestepEmbedSequential(*layers))

        self.out = nn.Sequential(
            # normalization(ch),
            normalization(ch, num_groups=num_norm_groups),
            SiLU(),
            conv_nd(dims, model_channels, out_channels, 3, padding=1),
        )
        if zero_out_last_layer:
            self.out[-1] = zero_module(self.out[-1])

    def input_sizes_on_scales(self, input_size, top_scale=None):
        """
        Returns:
            List of input_sizes on different scales.
        e.g. input(18, 18)
        [(18, 18), (9, 9), (5, 5), (3, 3), (1, 1), (0, 0),....]
        e.g.input(32, 36)
        [(32, 36), (16, 18), (8, 9), (4, 5), (2, 3), (1, 2), (0, 0),....]
        e.g.input(1, 35)
        [(1, 53), (1, 27), (1, 14),...]
        """
        outputs = [(0,0)] * 30  # Assume no more than 30 levels.
        def scale_size(input_size, scale_ind=0, outputs=outputs):
            outputs[scale_ind] = input_size
            if input_size[0] <= 1 and input_size[1] <= 1:
                return
            down_size = (np.ceil(input_size[0]/2).astype(int), np.ceil(input_size[1]/2).astype(int))
            scale_size(down_size, scale_ind+1, outputs)
        scale_size(input_size, 0, outputs)
        if top_scale:
            outputs = outputs[:top_scale]
        return outputs

    @property
    def inner_dtype(self):
        """
        Get the dtype used by the torso of the model.
        """
        return next(self.input_blocks.parameters()).dtype

    def forward(self, x, timesteps, y=None):
        """
        Apply the model to an input batch.

        :param x: an [N x C x ...] Tensor of inputs.
        :param timesteps: a 1-D batch of timesteps.
        :param y: an [N] Tensor of labels, if class-conditional.
        :return: an [N x C x ...] Tensor of outputs.
        """
        assert (y is not None) == (
            self.num_classes is not None
        ), "must specify y if and only if the model is class-conditional"

        hs = []
        if len(timesteps.shape)==0:
            timesteps=timesteps[None]

        emb = self.time_embed(timestep_embedding(timesteps, self.model_channels))

        if self.num_classes is not None:
            assert y.shape == (x.shape[0],)
            emb = emb + self.label_emb(y)

        h = x.type(self.inner_dtype)
        layer = 0
        for module in self.input_blocks:
            h = module(h, emb)
            # print(layer, h.shape)
            hs.append(h)
            layer += 1
        h = self.middle_block(h, emb)
        for module in self.output_blocks:
            cat_in = torch.cat([h, hs.pop()], dim=1)
            h = module(cat_in, emb)
            # print(layer, h.shape)
            layer -= 1
        h = h.type(x.dtype)
        h = self.out(h)

        return h
