import numpy as np
import pytest
from planet_diffusion.cache import ArrayCache, PredictionCache


def test_resume_uses_predictions_but_rejects_different_identity(tmp_path):
    calls = []
    def model(a, y, x):
        calls.append((y, x))
        return a*2
    a = np.ones((1, 16, 16), np.float32)
    first = PredictionCache(tmp_path, {"seed": 42}, model)
    expected = first(a, -8, 0)
    second = PredictionCache(tmp_path, {"seed": 42}, model)
    np.testing.assert_array_equal(second(a, -8, 0), expected)
    assert calls == [(-8, 0)]
    with pytest.raises(ValueError, match="different inputs"):
        PredictionCache(tmp_path, {"seed": 43}, model)


def test_array_cache_bounds_bytes_and_evicts_least_recently_used():
    cache = ArrayCache(32)
    a = np.arange(4,dtype=np.float32)
    cache.put('a',a)
    cache.put('b',a.copy())
    assert cache.get('a') is a  # Refresh a, making b oldest.
    cache.put('c',a.copy())
    assert cache.get('b') is None
    assert cache.get('a') is a
    assert cache.bytes == 32
    cache.put('a',np.ones(2,np.float32))
    assert cache.bytes == 24
    cache.put('large',np.ones(9,np.float32))
    assert cache.get('large') is None and cache.bytes == 24


def test_array_cache_does_not_retain_batch_parents():
    cache = ArrayCache(16)
    batch = np.ones((100,4),np.float32)
    cache.put('patch',batch[0])
    patch = cache.get('patch')
    assert patch.flags.owndata and not np.shares_memory(patch,batch)
    assert cache.bytes == patch.nbytes == 16
    disabled = ArrayCache(0)
    disabled.put('patch',patch)
    assert disabled.get('patch') is None and disabled.bytes == 0


@pytest.mark.parametrize('budget',[-1,1.5,True])
def test_invalid_array_cache_budget(budget):
    with pytest.raises(ValueError,match='byte budget'):
        ArrayCache(budget)
