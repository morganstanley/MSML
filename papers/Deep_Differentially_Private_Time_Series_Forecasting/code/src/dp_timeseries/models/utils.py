from copy import deepcopy

from opacus.layers.dp_multihead_attention import DPMultiheadAttention
from opacus.layers.dp_rnn import DPLSTM
from torch.nn import (LSTM, Module, ModuleList, MultiheadAttention,
                      TransformerEncoder, TransformerEncoderLayer)

from .dp_estimator import DPPyTorchLightningEstimator


def create_dp_estimator(
        original_estimator_name: str,
        estimator_kwargs: dict[str]) -> DPPyTorchLightningEstimator:
    """Instantiates f"DP"{original_estimator_name} with estimator_kwargs

    Args:
        original_estimator_name (_type_): Name of non-DP estimator class.
        estimator_kwargs (_type_): kwargs to use for __init__ of DP estimator.

    Returns:
        DPPyTorchLightningEstimator: The DP estimator
    """

    from . import (DPDeepAREstimator, DPDLinearEstimator,
                   DPITransformerEstimator, DPSimpleFeedForwardEstimator,
                   DPTemporalFusionTransformerEstimator, DPWaveNetEstimator)

    cls = locals()[f'DP{original_estimator_name}']
    return cls(**estimator_kwargs)


def create_dp_compatible_layer(
        original_layer: Module):
    """Creates pure python layers from CUDA-native layers.

    Weights are not copied.
    Has no effect when layer is not LSTM or MultiHeadAttention.

    Args:
        original_layer: Layer to make compatible to opacus.
    """

    if isinstance(original_layer, LSTM):
        return DPLSTM(
            input_size=original_layer.input_size,
            hidden_size=original_layer.hidden_size,
            num_layers=original_layer.num_layers,
            bias=original_layer.bias,
            batch_first=original_layer.batch_first,
            dropout=original_layer.dropout,
            bidirectional=original_layer.bidirectional,
            proj_size=original_layer.proj_size)

    elif isinstance(original_layer, MultiheadAttention):
        return DPMultiheadAttention(
            embed_dim=original_layer.embed_dim,
            num_heads=original_layer.num_heads,
            dropout=original_layer.dropout,
            bias=(original_layer.in_proj_bias is not None),
            add_bias_kv=(original_layer.bias_k is not None),
            add_zero_attn=original_layer.add_zero_attn,
            kdim=original_layer.kdim,
            vdim=original_layer.vdim,
            batch_first=original_layer.batch_first)

    elif isinstance(original_layer, TransformerEncoderLayer):
        encoder_layer = deepcopy(original_layer)
        encoder_layer.self_attn = create_dp_compatible_layer(encoder_layer.self_attn)
        return encoder_layer

    elif isinstance(original_layer, TransformerEncoder):
        encoder = deepcopy(original_layer)

        encoder.layers = ModuleList([
            create_dp_compatible_layer(encoder_layer)
            for encoder_layer in encoder.layers])

        return encoder

    else:
        return original_layer
