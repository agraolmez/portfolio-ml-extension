"""Storage activation requires reviewed engineering evidence, not a new model."""
import json
from pathlib import Path
from feature_geometry.history import sha


def verify_storage_approval(root, config_path, approval):
    root = Path(root).resolve()
    if approval.get('exact_overflow_disk_reuse') is not True:
        raise ValueError('Explicit full-run approval of exact overflow disk reuse is required')
    value = Path(approval.get('storage_check_path', ''))
    path = (root/value).resolve()
    if (not path.is_relative_to(root/'output') or not path.is_file()
            or sha(path) != approval.get('storage_check_sha256')):
        raise ValueError('A bound measured storage check is required')
    receipt = json.loads(path.read_text())
    if (receipt.get('status') != 'PASS' or receipt.get('config_sha256') != sha(config_path)
            or receipt.get('real_accepted_data') is not True
            or receipt.get('exact_roundtrip') is not True
            or receipt.get('parent_neutral_signals_exact') is not True
            or receipt.get('portfolio_evaluation_run') is not False
            or receipt.get('gain_optimization_run') is not False):
        raise ValueError('Storage equivalence or scientific identity check failed')
    inventory = receipt.get('code_sha256', {})
    required = {'temporal_numerics/spill.py', 'temporal_storage/cache.py'}
    if not required <= set(inventory):
        raise ValueError('Storage implementation is absent from the measured evidence')
    for relative, expected in inventory.items():
        source = (root/relative).resolve()
        if not source.is_relative_to(root) or sha(source) != expected:
            raise ValueError('Measured storage implementation changed')
    return receipt
