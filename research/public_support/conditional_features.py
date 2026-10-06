"""Public implementation of the documented E4 design [B, c1*B_J, ...].

Written for this source export. No external reference implementation is bundled.
The input features are already normalized; this function adds no normalization.
"""
from dataclasses import dataclass
import numbers
import numpy as np

@dataclass(frozen=True)
class InteractionDesign:
    design: np.ndarray
    columns: tuple
    parent_width: int
    state_count: int
    interaction_columns: tuple


def augment_parent_features(parent_features, state, *, parent_names, state_names,
                            interaction_columns):
    matrix = np.asarray(parent_features, dtype=np.float64)
    levels = np.asarray(state, dtype=np.float64)
    if matrix.ndim != 2 or levels.ndim != 1 or 0 in matrix.shape:
        raise ValueError('Expected a nonempty feature matrix and a state vector')
    if not np.all(np.isfinite(matrix)) or not np.all(np.isfinite(levels)):
        raise ValueError('Inputs must be finite')
    labels = tuple(parent_names)
    states = tuple(state_names)
    for names, size in ((labels, matrix.shape[1]), (states, levels.size)):
        if len(names) != size or len(set(names)) != size:
            raise ValueError('Each input column requires a distinct name')
    indices = tuple(interaction_columns)
    if any(isinstance(i, (bool, np.bool_)) or not isinstance(i, numbers.Integral)
           for i in indices):
        raise ValueError('Column selections must be integer indices')
    indices = tuple(map(int, indices))
    if len(set(indices)) != len(indices) or any(i < 0 or i >= matrix.shape[1] for i in indices):
        raise ValueError('Column selection is out of range or repeated')
    if levels.size and not indices:
        raise ValueError('An enabled state requires selected columns')
    names = labels + tuple(f'state[{s}]::{labels[i]}' for s in states for i in indices)
    if len(set(names)) != len(names):
        raise ValueError('Interaction names must remain distinct')
    answer = np.empty((matrix.shape[0], len(names)), dtype=np.float64)
    width = matrix.shape[1]
    answer[:, :width] = matrix
    for k in range(levels.size):
        start = width + k * len(indices)
        answer[:, start:start+len(indices)] = matrix[:, indices] * levels[k]
    if not np.all(np.isfinite(answer)):
        raise ValueError('State-feature multiplication overflowed')
    return InteractionDesign(answer, names, width, levels.size, indices)
