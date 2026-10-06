"""A proposed full-study time allowance; no numerical accuracy change."""
from copy import deepcopy
import json
from pathlib import Path
from temporal_numerics.optimizer import LBFGSBSettings
from temporal_numerics.pilot import validate_config


ROOT = Path(__file__).resolve().parents[1]


def pilot_configuration(config):
    parent = json.loads((ROOT/'config/extension3_study_v2.json').read_text())
    original = json.loads((ROOT/'config/extension3_study_v1.json').read_text())
    validate_config(parent, original)
    expected = deepcopy(parent)
    expected['id'] = 'extension3-temporal-study-v3-proposal-20260919'
    expected['status'] = 'PROPOSED_FULL_RUN_TIME_ALLOWANCE_NOT_AUTHORIZATION'
    expected['optimizer']['wall_seconds'] = 10800.0
    if config != expected:
        raise ValueError('Only the proposed three-hour per-fit allowance may differ from the successful pilot')
    return parent


def settings(config):
    pilot_configuration(config)
    return LBFGSBSettings(**config['optimizer'])
