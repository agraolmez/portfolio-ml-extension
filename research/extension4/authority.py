"""The proposal is never permission; user authority must be separately recorded."""
import json
from pathlib import Path
from feature_geometry.history import sha


def require_approval(root, config_path, approval_path):
    root = Path(root).resolve()
    config_path, approval_path = Path(config_path).resolve(), Path(approval_path).resolve()
    if not config_path.is_relative_to(root) or not approval_path.is_relative_to(root):
        raise PermissionError('Study configuration and approval must be in the authorized project')
    if not approval_path.is_file():
        raise PermissionError('Economic pilot and full run require the pending consolidated user approval')
    config = json.loads(config_path.read_text())
    approval = json.loads(approval_path.read_text())
    if (approval.get('status') != 'APPROVED_BY_USER' or approval.get('end_to_end') is not True
            or not approval.get('user_message') or not approval.get('recorded_at_utc')
            or approval.get('paid_resources') is not False
            or approval.get('config_sha256') != sha(config_path)
            or approval.get('scientific_exceptions') != config['scientific_exceptions_requiring_approval']):
        raise PermissionError('Missing, incomplete or mismatched end-to-end approval')
    plan = root/config['paths']['study_plan']
    if approval.get('study_plan_sha256') != sha(plan):
        raise PermissionError('Approved study plan changed')
    if (config['compute_budget']['paid'] is not False or not config['compute_budget']['local_only']
            or config['history']['seeds'] != [1, 2, 3] or config['history']['risk_aversion'] != 10.):
        raise ValueError('Unsupported protocol; do not silently broaden the implemented study')
    return config, approval
