"""Research-inspired multi-scale context applied to the existing native U-Net.

This is a small dilated/global-context residual adapter, not a reproduction of
MORDEN, a Transformer, or deformable large-kernel attention. The original
encoder, decoder, output heads and 2048/native coordinate convention stay intact.
External evaluation data are never inputs to this module's training helpers.
"""
from collections.abc import Mapping

import torch
from torch import nn
from torch.nn import functional as F


METHODS = ('control', 'multiscale_context')


def _normalization_groups(channels):
    # At least two channels per group, including a pooled B=1,H=W=1 branch.
    for groups in range(min(16, channels // 2), 0, -1):
        if channels % groups == 0:
            return groups
    return 1


class MultiScaleResidualContext(nn.Module):
    """Parallel depthwise dilations and crop-global pooling with a zero gate.

    A zero gate makes the initial adapter exactly the identity. It is still
    evaluated so the gate can receive a gradient on the first optimizer update.
    GroupNorm avoids batch-size-dependent pooled BatchNorm failures. No extra
    Dropout is introduced into a paired training RNG stream.
    """
    def __init__(self, channels, hidden=128, rates=(1, 3, 5)):
        super().__init__()
        if not isinstance(channels, int) or isinstance(channels, bool) or channels < 2:
            raise ValueError('channels must be an integer >=2')
        if not isinstance(hidden, int) or isinstance(hidden, bool) or not 2 <= hidden <= 128:
            raise ValueError('hidden must be an integer in [2,128]')
        rates = tuple(rates)
        if not rates or any(
                not isinstance(rate, int) or isinstance(rate, bool) or rate <= 0
                for rate in rates) or len(set(rates)) != len(rates):
            raise ValueError('dilation rates must be unique positive integers')
        hidden = min(hidden, channels)
        self.channels, self.hidden, self.rates = channels, hidden, rates
        groups = _normalization_groups(hidden)
        self.reduce = nn.Sequential(nn.Conv2d(channels, hidden, 1, bias=False),
                                    nn.GroupNorm(groups, hidden), nn.SiLU())
        self.branches = nn.ModuleList([
            nn.Sequential(nn.Conv2d(hidden, hidden, 3, padding=rate, dilation=rate,
                                    groups=hidden, bias=False),
                          nn.GroupNorm(groups, hidden), nn.SiLU())
            for rate in rates])
        self.global_branch = nn.Sequential(nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(hidden, hidden, 1, bias=False), nn.GroupNorm(groups, hidden), nn.SiLU())
        self.fuse = nn.Conv2d(hidden * (len(rates) + 1), channels, 1, bias=False)
        self.gate = nn.Parameter(torch.zeros(()))

    def forward(self, features):
        if not isinstance(features,torch.Tensor) or features.ndim != 4 or features.shape[1] != self.channels:
            raise ValueError(f'Expected B,{self.channels},H,W features')
        reduced = self.reduce(features)
        values = [branch(reduced) for branch in self.branches]
        values.append(F.interpolate(self.global_branch(reduced),size=reduced.shape[-2:],
                                    mode='bilinear',align_corners=False))
        residual = self.fuse(torch.cat(values,dim=1))
        return features + self.gate.tanh() * residual


class PaperUNet(nn.Module):
    """Wrap the reviewed SMP0.5 U-Net with control or one bottleneck change."""
    def __init__(self, base, method='control', hidden=128, rates=(1, 3, 5)):
        super().__init__()
        if method not in METHODS:
            raise ValueError(f'Unknown method: {method}; choose {METHODS}')
        if getattr(base,'classification_head',None) is not None:
            raise ValueError('The native experiment supports segmentation heads only')
        if not all(hasattr(base,name) for name in ('encoder','decoder','segmentation_head','check_input_shape')):
            raise ValueError('Expected the reviewed SMP U-Net interface')
        channels = getattr(base.encoder,'out_channels',None)
        if not channels or not isinstance(channels[-1],int) or channels[-1] < 2:
            raise ValueError('Encoder must declare valid output channels')
        self.base, self.method = base, method
        # Control/context construction consumes the same surrounding RNG stream.
        with torch.random.fork_rng(devices=[]):
            self.context = (MultiScaleResidualContext(channels[-1],hidden,rates)
                            if method == 'multiscale_context' else nn.Identity())

    @property
    def encoder(self):
        return self.base.encoder

    def forward(self, image):
        if not isinstance(image,torch.Tensor) or image.ndim != 4 or image.shape[1] != 3:
            raise ValueError('Expected B,3,H,W normalized native images')
        self.base.check_input_shape(image)
        features = list(self.base.encoder(image))
        features[-1] = self.context(features[-1])
        # SMP0.5 decoder takes the feature LIST, not decoder(*features).
        decoded = self.base.decoder(features)
        return self.base.segmentation_head(decoded)


def load_base_weights(model, checkpoint_or_state):
    """Strict warm start from the original native checkpoint's base weights.

    Full PaperUNet resumes must use model.load_state_dict instead, preserving
    the trained adapter rather than accidentally dropping it.
    """
    if not isinstance(model,PaperUNet) or not isinstance(checkpoint_or_state,Mapping):
        raise ValueError('A PaperUNet and native checkpoint/state mapping are required')
    state = checkpoint_or_state.get('model_state_dict',checkpoint_or_state)
    if not isinstance(state,Mapping) or not state:
        raise ValueError('Missing native model_state_dict')
    if any(str(key).startswith(('base.','context.')) for key in state):
        raise ValueError('For a complete PaperUNet resume, load the complete model state instead of base weights')
    return model.base.load_state_dict(state,strict=True)


def build_from_native_checkpoint(checkpoint, method='control', encoder=None,
                                 hidden=128, rates=(1, 3, 5)):
    """Build from explicit native checkpoint bytes; never download pretraining."""
    if not isinstance(checkpoint,Mapping) or 'model_state_dict' not in checkpoint:
        raise ValueError('An original native checkpoint with model_state_dict is required')
    if method not in METHODS:
        raise ValueError(f'Unknown method: {method}; choose {METHODS}')
    config = checkpoint.get('config',{})
    if not isinstance(config,Mapping):
        raise ValueError('Checkpoint config must be a mapping')
    saved_encoder = config.get('encoder','tu-convnext_tiny')
    encoder = saved_encoder if encoder is None else encoder
    if encoder != saved_encoder:
        raise ValueError('Requested encoder differs from the warm-start checkpoint')
    import segmentation_models_pytorch as smp
    base = smp.Unet(encoder_name=encoder,encoder_weights=None,in_channels=3,
                    classes=2,activation=None)
    model = PaperUNet(base,method,hidden,rates)
    load_base_weights(model,checkpoint)
    return model
