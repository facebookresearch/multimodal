from torchmultimodal.architectures.omnivore import OmnivoreArchitecture
from torchmultimodal.modules.encoders.swin_transformer_3d_encoder import SwinTransformer3dEncoder, PatchEmbedOmnivore

from torch import nn


def _imagenet1k_head(input_dim: int) -> nn.Module:
    return nn.Linear(input_dim, 1000, bias=True)

def _kinetics400_head(input_dim: int) -> nn.Module:
    return nn.Sequential(
        nn.Dropout(p=0.5),
        nn.Linear(input_dim, 400, bias=True),
    )

def _sunrgbd_head(input_dim: int) -> nn.Module:
    return nn.Linear(input_dim, 19, bias=True)

def _multimodal_head(input_dim: int) -> nn.Module:
    return nn.ModuleDict({
        "image": _imagenet1k_head(input_dim),
        "rgbd": _sunrgbd_head(input_dim),
        "video": _kinetics400_head(input_dim),
    })

def omnivore_swin_t() -> nn.Module:
    embed_dim = 96
    norm_layer = nn.LayerNorm
    patch_embed = PatchEmbedOmnivore(
        patch_size=(2, 4, 4),
        embed_dim=embed_dim,
        norm_layer=norm_layer,
    )
    encoder = SwinTransformer3dEncoder(
        embed_dim=embed_dim,
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        window_size=(8, 7, 7),
        stochastic_depth_prob=0.2,
        norm_layer=norm_layer,
        patch_embed=patch_embed,
    )
    heads = _multimodal_head(input_dim=encoder.num_features)
    return OmnivoreArchitecture(encoder, heads)


