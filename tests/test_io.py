import inspect
import numpy as np

from timex import fit, io


def test_read_generic_plain_file_returns_none_design_matrix(synthetic_lc):
    x, y, yerr, X, texp, x_hr, ref_time = io.read_generic(
        synthetic_lc, binsize=None, verbose=False)
    assert X is None
    assert x.shape == y.shape == yerr.shape
    assert x_hr.shape == (500,)


def test_read_generic_trend_with_bias_has_no_duplicate_constant_column(synthetic_lc):
    _, _, _, X, _, _, _ = io.read_generic(
        synthetic_lc, binsize=None, trend=1, add_bias=True, verbose=False)
    assert X is not None
    # a duplicated constant column makes the matrix rank deficient
    assert np.linalg.matrix_rank(X) == X.shape[1]


def test_read_generic_trend_without_bias_has_no_constant_column(synthetic_lc):
    _, _, _, X, _, _, _ = io.read_generic(
        synthetic_lc, binsize=None, trend=1, add_bias=False, verbose=False)
    assert np.linalg.matrix_rank(X) == X.shape[1]
    # no column may be constant
    assert not np.any(X.std(axis=0) == 0)


def test_read_generic_bias_only_gives_one_constant_column(synthetic_lc):
    _, _, _, X, _, _, _ = io.read_generic(
        synthetic_lc, binsize=None, add_bias=True, verbose=False)
    assert X.shape[1] == 1
    assert np.allclose(X[:, 0], 1.0)


def test_read_generic_chunk_offset_without_covariates(synthetic_lc):
    _, _, _, X, _, _, _ = io.read_generic(
        synthetic_lc, binsize=None, chunk_offset=True, chunk_thresh=0.02,
        verbose=False)
    # contiguous data is a single chunk
    assert X.shape[1] == 1
    assert np.allclose(X[:, 0], 1.0)


def test_read_generic_with_covariates(synthetic_lc_aux):
    _, _, _, X, _, _, _ = io.read_generic(
        synthetic_lc_aux, binsize=None, verbose=False)
    assert X.shape[1] == 2
    # covariates are standardized
    assert np.allclose(X.mean(axis=0), 0, atol=1e-8)


def test_read_generic_covariates_with_trend_and_bias(synthetic_lc_aux):
    _, _, _, X, _, _, _ = io.read_generic(
        synthetic_lc_aux, binsize=None, trend=2, add_bias=True, verbose=False)
    assert np.linalg.matrix_rank(X) == X.shape[1]


def test_fit_chunk_thresh_default_matches_io_signature():
    io_default = inspect.signature(io.read_generic).parameters['chunk_thresh'].default
    assert fit.defaults['data']['chunk_thresh'] == io_default


def test_chunk_offset_with_fit_default_does_not_split_every_point(synthetic_lc):
    _, _, _, X, _, _, _ = io.read_generic(
        synthetic_lc, binsize=None, chunk_offset=True,
        chunk_thresh=fit.defaults['data']['chunk_thresh'], verbose=False)
    assert X.shape[1] == 1
