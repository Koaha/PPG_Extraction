import pytest
from RRest.estimate_rr.AR_RR import get_rr

def test_get_rr():
    mock_sig = [1]*100
    res = get_rr(mock_sig,10)
    assert res > 0
    pass