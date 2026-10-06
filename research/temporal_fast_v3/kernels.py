"""Same double-precision formulas with fewer temporary array passes."""
import numpy as np
from temporal_fast.backend import sincos
from temporal_fast.streaming import feature_parts as previous_parts
from temporal_study.controls import CurrentOnlyFeatureMap
from . import native


def feature_parts(fmap,z,vol,innovation,gains,trig_backend='accelerate'):
    if trig_backend!='accelerate' or np.all(gains==1.):
        return previous_parts(fmap,z,vol,innovation,gains,trig_backend)
    inputs=fmap.inputs(z,innovation,gains);vol=np.asarray(vol)
    if vol.shape!=(len(inputs),) or not np.isfinite(vol).all() or np.any(vol<=0):
        raise ValueError('Invalid volatility')
    with np.errstate(over='raise',invalid='raise'):
        scale=np.exp(fmap.theta);angles=(inputs*scale)@fmap.parent.frequencies
    cosine,sine=sincos(angles)
    normalized,norm=native.normalize(cosine,sine)
    return inputs,vol,scale,cosine,sine,normalized,norm


def signal_from_parts(parts):
    return native.signals(parts[5],parts[1])


def rank_one_gradient(fmap,z,innovation,parts,adjoint,beta):
    _,vol,scale,cosine,sine,normalized,norm=parts
    adjoint,beta=np.asarray(adjoint),np.asarray(beta)
    if (adjoint.shape!=(len(z),) or beta.shape!=(fmap.parent.columns,)
            or not np.isfinite(adjoint).all() or not np.isfinite(beta).all()):
        raise ValueError('Finite aligned rank-one adjoint required')
    a=adjoint/vol
    angle_gradient=native.angle_adjoint(a,normalized,normalized.T@a,beta[1:]/norm,cosine,sine)
    input_gradient=angle_gradient@fmap.parent.frequencies.T
    derivative=z if type(fmap) is CurrentOnlyFeatureMap else innovation.values
    by_characteristic=native.characteristic_adjoint(input_gradient,scale,derivative)
    return np.bincount(fmap.groups,weights=by_characteristic,minlength=fmap.gain_count)
