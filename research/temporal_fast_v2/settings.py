"""Further execution changes; scientific settings and resource limits retained."""
from types import MappingProxyType
from temporal_fast.settings import EXECUTION as PREVIOUS,environment_identity

EXECUTION=MappingProxyType(dict(PREVIOUS,parallel_integrity_checks=True,
    rank_one_adjoint=True,shared_horner=True,packed_horner=False,
    neutral_operators='original independent Horner recursions',
    additional_persistent_cache_bytes=0))

VARIANTS=MappingProxyType(dict(parallel_checks=True,rank_one=True,packed=False,shared=True))
