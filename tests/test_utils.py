import numpy as np
import pytest
import scipy as sp
import theano
import theano.tensor as tt

from pymc3_hmm.utils import (
    compute_trans_freqs,
    logdotexp,
    multilogit_inv,
    tt_logdotexp,
    tt_logsumexp,
)


def test_compute_trans_freqs():
    res = compute_trans_freqs(np.r_[0, 1, 1, 1, 1, 0, 1], 2, counts_only=True)
    assert np.array_equal(res, np.c_[[0, 1], [2, 3]])


@pytest.mark.parametrize(
    "test_input",
    [
        -10,
        np.array([0.0, -10.0, 1e4]),
        np.array(
            [
                [[0.95557887], [0.88492326]],
                [[0.27770323], [0.73042471]],
                [[0.59677073], [0.22220477]],
                [[0.39335336], [0.83246557]],
            ]
        ),
    ],
)
def test_logsumexp(test_input):
    np_res = sp.special.logsumexp(test_input)
    tt_res = tt_logsumexp(tt.as_tensor_variable(test_input)).eval()
    assert np.array_equal(np_res, tt_res)


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


@theano.config.change_flags(compute_test_value="warn")
@np.errstate(over="ignore", under="ignore")
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


@pytest.mark.parametrize(
    "test_input, test_output",
    [
        (np.array([[0], [0]]), np.array([[0.5, 0.5], [0.5, 0.5]])),
        (np.array([[-10], [10]]), np.array([[0.0, 1.0], [1.0, 0.0]])),
        (
            np.array([[-5e10, -5e10], [1, -10]]),
            np.array([[0.0, 0.0, 1.0], [0.73, 0.0, 0.27]]),
        ),
        (
            np.array([[5e10, 5e10], [1, -10], [-5e10, 5e10]]),
            np.array([[0.5, 0.5, 0.0], [0.73, 0.0, 0.27], [0.0, 1.0, 0.0]]),
        ),
    ],
)
def test_multilogit_inv(test_input, test_output):

    # NumPy testing
    res = multilogit_inv(test_input)
    assert np.array_equal(res.round(2), test_output)

    # Theano testing
    res = multilogit_inv(tt.as_tensor_variable(test_input))
    res = res.eval()
    assert np.array_equal(res.round(2), test_output)
