"""Same immutable format and reconstruction checks with a batch source verifier."""
import json
import numpy as np
from feature_geometry.history import sha
from feature_geometry.moments import MarketMoment,TrainingSample
from geometry_study.kernel import LeanBasis
from temporal_extension.reference import Innovation,month_key
from temporal_numerics.spill import OverflowStore as PreviousStore,payload_fingerprint


class OverflowStore(PreviousStore):
    def __init__(self,*args,verify_inputs=None,**kwargs):
        super().__init__(*args,**kwargs)
        if verify_inputs is not None and not callable(verify_inputs):raise ValueError('Batch input verifier must be callable')
        self.verify_inputs=verify_inputs

    def get(self, day, cutoff):
        path, commit = self._paths(day)
        month_key(cutoff)
        if not path.exists() and not commit.exists():
            return None
        if not path.is_file() or not commit.is_file():
            raise ValueError('Partial overflow artifact requires review')
        authority = json.loads(commit.read_text())
        if (authority.get('context') != self.context or authority.get('decision_date') != day
                or sha(path) != authority.get('payload_sha256')):
            raise ValueError('Overflow source identity or payload changed')
        with np.load(path, allow_pickle=False) as data:
            metadata = json.loads(str(data['metadata'].item()))
            if (metadata.get('context') != self.context or metadata.get('decision_date') != day
                    or metadata['available_at'] > cutoff):
                raise ValueError('Overflow identity or outcome maturity differs')
            if self.verify_inputs is not None:self.verify_inputs(metadata['accepted_inputs_sha256'])
            else:
                for relative, expected in metadata['accepted_inputs_sha256'].items():
                    self.verify_input(relative, expected)
            market = MarketMoment(day, data['ids'], tuple(data[f'z{i}'] for i in range(13)),
                tuple(data[f'vol{i}'] for i in range(13)), tuple(data[f'growth{i}'] for i in range(13)),
                data['m'], data['covariance'], data['impact'], metadata['wealth'], metadata['gamma'],
                metadata['market_source_id'])
            sample = TrainingSample(market, data['returns'], metadata['available_at'])
            basis = LeanBasis.__new__(LeanBasis)
            basis.sample = sample
            basis.current_lu = (data['current_lu'].copy(), data['current_pivots'].copy())
            basis.prior_lu = (data['prior_lu'].copy(), data['prior_pivots'].copy())
            innovations = tuple(Innovation(data[f'innovation{i}'], data[f'valid{i}'], metadata['reasons'][i])
                                for i in range(13))
        n = len(market.ids)
        for lu, pivots in (basis.current_lu, basis.prior_lu):
            if lu.shape != (n,n) or pivots.shape != (n,) or pivots.dtype.kind not in 'iu' or not np.isfinite(lu).all():
                raise ValueError('Invalid restored economic operator')
            lu.setflags(write=False); pivots.setflags(write=False)
        if (market.fingerprint != metadata['market_fingerprint'] or sample.fingerprint != metadata['sample_fingerprint']
                or payload_fingerprint(basis, innovations) != metadata['payload_fingerprint']
                or metadata['payload_fingerprint'] != authority['payload_fingerprint']):
            raise ValueError('Restored operator content differs from its immutable payload')
        return basis, innovations
