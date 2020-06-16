import numpy as np

import theano.tensor as tt

from pymc3_hmm.utils import compute_trans_freqs, logdotexp, tt_logdotexp


def test_compute_trans_freqs():
    res = compute_trans_freqs(np.r_[0, 1, 1, 1, 1, 0, 1], 2, counts_only=True)
    assert np.array_equal(res, np.c_[[0, 1], [2, 3]])


def test_logdotexp():
    A = np.c_[[1.0, 2.0], [3.0, 4.0], [10.0, 20.0]]
    b = np.c_[[0.1], [0.2], [30.0]].T

    test_res = logdotexp(np.log(A), np.log(b))
    assert test_res.shape == (2, 1)
    assert np.allclose(A.dot(b), np.exp(test_res))

    b = np.r_[0.1, 0.2, 30.0]
    test_res = logdotexp(np.log(A), np.log(b))
    assert test_res.shape == (2,)
    assert np.allclose(A.dot(b), np.exp(test_res))

    A = np.c_[[1.0, 2.0], [10.0, 20.0]]
    b = np.c_[[0.1], [0.2]].T
    test_res = logdotexp(np.log(A), np.log(b))
    assert test_res.shape == (2, 1)
    assert np.allclose(A.dot(b), np.exp(test_res))

    b = np.r_[0.1, 0.2]
    test_res = logdotexp(np.log(A), np.log(b))
    assert test_res.shape == (2,)
    assert np.allclose(A.dot(b), np.exp(test_res))


def test_tt_logdotexp():

    A = np.c_[[1.0, 2.0], [3.0, 4.0], [10.0, 20.0]]
    b = np.c_[[0.1], [0.2], [30.0]].T
    A_tt = tt.as_tensor_variable(A)
    b_tt = tt.as_tensor_variable(b)
    test_res = tt_logdotexp(tt.log(A_tt), tt.log(b_tt)).eval()
    assert test_res.shape == (2, 1)
    assert np.allclose(A.dot(b), np.exp(test_res))

    b = np.r_[0.1, 0.2, 30.0]
    test_res = tt_logdotexp(tt.log(A), tt.log(b)).eval()
    assert test_res.shape == (2,)
    assert np.allclose(A.dot(b), np.exp(test_res))

    A = np.c_[[1.0, 2.0], [10.0, 20.0]]
    b = np.c_[[0.1], [0.2]].T
    test_res = tt_logdotexp(tt.log(A), tt.log(b)).eval()
    assert test_res.shape == (2, 1)
    assert np.allclose(A.dot(b), np.exp(test_res))

    b = np.r_[0.1, 0.2]
    test_res = tt_logdotexp(tt.log(A), tt.log(b)).eval()
    assert test_res.shape == (2,)
    assert np.allclose(A.dot(b), np.exp(test_res))
