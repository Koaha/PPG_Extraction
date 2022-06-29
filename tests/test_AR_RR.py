# import pytest
import numpy as np
from RRest.estimate_rr.AR_RR import get_rr

def test_get_rr():
    mock_sig = np.random.random(10000)
    # res = get_rr(mock_sig,10)
    res = -1
    assert res > 0
    pass