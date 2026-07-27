import inspect
import numpy as np
import pytest

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


def test_fit_default_chunk_thresh_is_a_real_gap():
    """Pinned against a literal, not against io's default.

    test_fit_chunk_thresh_default_matches_io_signature only checks the two
    defaults against each other, so moving both back to 0 together would
    satisfy it while making every point its own chunk.
    """
    assert fit.defaults['data']['chunk_thresh'] > 0


@pytest.mark.parametrize('thresh', [0, 0.0, -1.0, None])
def test_read_generic_rejects_non_positive_chunk_thresh(synthetic_lc, thresh):
    """np.diff(x) > 0 is true everywhere, so a zero threshold appends an N x N
    identity to the design matrix: a perfect fit that erases the transit."""
    with pytest.raises(ValueError, match='chunk_thresh') as excinfo:
        io.read_generic(synthetic_lc, binsize=None, chunk_offset=True,
                        chunk_thresh=thresh, verbose=False)
    assert repr(thresh) in str(excinfo.value)


def test_non_positive_chunk_thresh_is_ignored_without_chunk_offset(synthetic_lc):
    """The threshold is unused unless chunk_offset asks for the columns, so
    the guard must not reject configurations it cannot affect."""
    _, _, _, X, _, _, _ = io.read_generic(
        synthetic_lc, binsize=None, chunk_offset=False, chunk_thresh=0,
        verbose=False)
    assert X is None


@pytest.mark.parametrize('trim_beg,trim_end', [(1.0, None), (None, 1.0)])
def test_read_generic_rejects_trim_that_removes_every_point(
        synthetic_lc, trim_beg, trim_end):
    """The data spans 0.1 d, so trimming a day off either end empties it.

    Without the guard this surfaces as numpy's 'zero-size array to reduction
    operation minimum' from inside np.linspace, which names neither parameter.
    """
    with pytest.raises(ValueError, match='trim_beg') as excinfo:
        io.read_generic(synthetic_lc, binsize=None, trim_beg=trim_beg,
                        trim_end=trim_end, verbose=False)
    message = str(excinfo.value)
    assert 'trim_end' in message
    assert '0 points' in message


def test_read_generic_keeps_points_a_partial_trim_leaves(synthetic_lc):
    """The control: a trim that only removes part of the series still works,
    so the guard cannot be satisfied by rejecting every trim."""
    x, _, _, _, _, _, _ = io.read_generic(
        synthetic_lc, binsize=None, trim_beg=0.02, trim_end=0.02, verbose=False)
    assert 0 < x.size < 120
