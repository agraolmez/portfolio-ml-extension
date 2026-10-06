"""Selected measured computations; unchanged science and resource ceilings."""
from types import MappingProxyType
from temporal_fast_v3.settings import EXECUTION as PREVIOUS, VARIANTS as PREVIOUS_VARIANTS
from temporal_fast_v3.settings import environment_identity as previous_environment
from .native import identity
from .source_checks import identity as source_identity

EXECUTION=MappingProxyType(dict(PREVIOUS,compact_evaluation_reuse_maximum_bytes=8*2**30,
    intermediate_reuse_lifetime='one objective and gradient call; never across parameters',
    intermediate_reuse_headroom_bytes=4*2**30,intermediate_reuse_process_admission_ceiling_bytes=20*2**30,
    source_panel_memory_maximum_bytes=3*2**30,source_panels_released_after_preparation_pass=True,
    batched_accepted_file_checks=True,deferred_overflow_loading=True,vectorized_calendar_reference=True,
    overflow_workers_share_numerical_pool=True))
VARIANTS=MappingProxyType(dict(PREVIOUS_VARIANTS,reuse_bytes=8*2**30,reuse_mode='trig',deferred_loads=True))


def environment_identity():
    return dict(previous_environment(),compact_build=identity(),source_checks_build=source_identity())
