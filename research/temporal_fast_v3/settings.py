"""Strict-double compiled kernels; original science and resource limits."""
from types import MappingProxyType
from temporal_fast_v2.settings import EXECUTION as PREVIOUS, VARIANTS as PREVIOUS_VARIANTS
from temporal_fast_v2.settings import environment_identity as previous_environment
from .native import identity

EXECUTION=MappingProxyType(dict(PREVIOUS,fused_feature_arrays=True,fused_adjoint_arrays=True,
    native_arithmetic='IEEE binary64; fast math disabled; FMA contraction disabled',
    neutral_native_kernels=False))
VARIANTS=MappingProxyType(dict(PREVIOUS_VARIANTS,fused_parts=True,fused_gradient=True))


def environment_identity():
    return dict(previous_environment(),native_build=identity())
