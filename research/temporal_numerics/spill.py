"""Exact disk reuse for immutable overflow operators.

No history is opened and no experiment runs on import. Every restored payload
is bound to source/reference/study identities, its accepted inputs and maturity.
This store is not used by the approved pilot. Its separate full-study entry point
requires explicit approval and a successful measured roundtrip check.
"""
import json
import os
from pathlib import Path
import tempfile
import numpy as np
from feature_geometry.features import fingerprint
from feature_geometry.history import sha
from feature_geometry.moments import MarketMoment, TrainingSample
from geometry_study.kernel import LeanBasis
from temporal_extension.reference import Innovation, month_key


def payload_fingerprint(basis, innovations):
    return fingerprint(basis.sample.fingerprint, *basis.current_lu, *basis.prior_lu,
                       *(a for x in innovations for a in (x.values, x.valid)),
                       json.dumps([x.reasons for x in innovations], sort_keys=True, allow_nan=False))


class OverflowStore:
    def __init__(self, directory, *, source_fingerprint, reference_fingerprint,
                 study_lock_sha256, maximum_bytes, verify_input):
        if (type(maximum_bytes) is not int or maximum_bytes <= 0 or not callable(verify_input)
                or any(not isinstance(x,str) or not x for x in
                       (source_fingerprint, reference_fingerprint, study_lock_sha256))):
            raise ValueError('Explicit identities, disk bound and input verifier are required')
        self.directory = Path(directory)
        self.maximum_bytes, self.verify_input = maximum_bytes, verify_input
        self.context = dict(format_version=2, source_fingerprint=source_fingerprint,
                            reference_fingerprint=reference_fingerprint, study_lock_sha256=study_lock_sha256)

    def _paths(self, day):
        month_key(day)
        # Canonical ISO spelling prevents aliases/traversal of cache keys.
        from datetime import date
        if date.fromisoformat(day).isoformat() != day:
            raise ValueError('A canonical calendar-month key is required')
        return self.directory/(day+'.npz'), self.directory/(day+'.json')

    def put(self, day, basis, innovations, accepted_inputs):
        path, commit = self._paths(day)
        if (type(basis) is not LeanBasis or basis.sample.market.decision_date != day
                or len(innovations) != 13 or not accepted_inputs):
            raise ValueError('A dated unchanged basis, thirteen innovations and accepted inputs are required')
        for relative, expected in accepted_inputs.items():
            self.verify_input(relative, expected)
        if path.exists() or commit.exists():
            raise FileExistsError('Overflow artifacts are immutable')
        m = basis.sample.market
        arrays = dict(ids=m.ids, m=m.m, covariance=m.covariance, impact=m.impact,
                      returns=basis.sample.returns, current_lu=basis.current_lu[0],
                      current_pivots=basis.current_lu[1], prior_lu=basis.prior_lu[0],
                      prior_pivots=basis.prior_lu[1])
        for i in range(13):
            arrays.update({f'z{i}':m.characteristics[i], f'vol{i}':m.volatility[i],
                           f'growth{i}':m.growth[i], f'innovation{i}':innovations[i].values,
                           f'valid{i}':innovations[i].valid})
        metadata = dict(context=self.context, decision_date=day, available_at=basis.sample.available_at,
            wealth=m.wealth, gamma=m.gamma, market_source_id=m.source_id,
            market_fingerprint=m.fingerprint, sample_fingerprint=basis.sample.fingerprint,
            reasons=[x.reasons for x in innovations], accepted_inputs_sha256=dict(accepted_inputs),
            payload_fingerprint=payload_fingerprint(basis, innovations))
        arrays['metadata'] = np.array(json.dumps(metadata, sort_keys=True, allow_nan=False))
        self.directory.mkdir(parents=True, exist_ok=True)
        used = sum(p.stat().st_size for p in self.directory.iterdir() if p.is_file())
        if used + sum(a.nbytes for a in arrays.values()) > self.maximum_bytes:
            raise OSError('Overflow disk budget is exhausted')
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(prefix='.pending-', suffix='.npz', dir=self.directory, delete=False) as f:
                temporary = Path(f.name)
                np.savez(f, **arrays)
            digest = sha(temporary)
            body = dict(context=self.context, decision_date=day, payload_sha256=digest,
                        payload_fingerprint=metadata['payload_fingerprint'])
            text = json.dumps(body, indent=2)+'\n'
            if used + temporary.stat().st_size + len(text.encode()) > self.maximum_bytes:
                raise OSError('Overflow disk budget is exhausted')
            # A same-directory hard link atomically refuses an existing target.
            os.link(temporary, path)
            with commit.open('x') as f:
                f.write(text)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
        return body

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
