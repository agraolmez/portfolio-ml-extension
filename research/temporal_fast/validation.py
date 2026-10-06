"""Bind an optimized full run to completed numerical and environment evidence."""
import json
from pathlib import Path
from feature_geometry.history import sha
from .settings import EXECUTION,environment_identity


def verify_optimized_approval(root, approval):
    root=Path(root).resolve()
    config_path=root/'config/extension3_speed_v1.json'
    config=json.loads(config_path.read_text())
    if (approval.get('optimized_profile_engine') is not True
            or approval.get('optimization_config_sha256')!=sha(config_path)
            or config['execution']!=dict(EXECUTION)):
        raise ValueError('Full-run approval must bind the verified optimized execution configuration')
    path=(root/approval.get('speed_verification_path','')).resolve()
    if (not path.is_relative_to(root/'output') or not path.is_file()
            or sha(path)!=approval.get('speed_verification_sha256')):
        raise ValueError('A bound complete speed-verification receipt is required')
    receipt=json.loads(path.read_text())
    if (receipt.get('status')!='VERIFIED_OPTIMIZATION_ONLY'
            or receipt.get('execution_settings')!=dict(EXECUTION)
            or receipt.get('execution_environment')!=environment_identity()
            or receipt.get('full_pilot_profiles_passed') is not True
            or receipt.get('both_neutral_coefficients_match_parent') is not True
            or receipt.get('both_saved_pilots_still_converged') is not True
            or receipt.get('portfolio_evaluation_run') is not False
            or receipt.get('gain_optimization_run') is not False):
        raise ValueError('Optimized neutral, convergence or runtime-environment evidence is incomplete')
    inventory=receipt.get('code_sha256',{})
    if not {'temporal_fast/backend.py','temporal_fast/streaming.py','temporal_fast/training.py'}<=set(inventory):
        raise ValueError('Optimized implementation missing from verification inventory')
    for name,expected in inventory.items():
        source=(root/name).resolve()
        if not source.is_relative_to(root) or sha(source)!=expected:
            raise ValueError('Verified optimized implementation changed')
    return receipt
