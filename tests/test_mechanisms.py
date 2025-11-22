import numpy as np
import pytest

from causalexplain.generators import mechanisms


@pytest.fixture(autouse=True)
def _seed():
    np.random.seed(0)


def test_gmm_cause_range():
    samples = mechanisms.gmm_cause(10, k=3, verbose=True)
    assert len(samples) == 10
    assert ((samples >= -1) & (samples <= 1)).all()


def test_linear_mechanism_zero_parents():
    mech = mechanisms.LinearMechanism(ncauses=0, points=4)
    output = mech(np.empty((4, 0)))
    assert output.shape == (4, 1)


def test_polynomial_mechanism_combines_causes():
    mech = mechanisms.Polynomial_Mechanism(ncauses=1, points=6, verbose=True)
    causes = np.ones((6, 1))
    result = mech(causes, verbose=True)
    assert result.shape == (6, 1)
    assert not np.allclose(result, 0)


def test_sigmoid_mechanisms_shapes():
    causes = np.ones((5, 2))
    sig_add = mechanisms.SigmoidAM_Mechanism(2, 5)
    sig_mix = mechanisms.SigmoidMix_Mechanism(2, 5)
    assert sig_add(causes).shape == (5, 1)
    assert sig_mix(causes).shape == (5, 1)


def test_gaussian_process_add_transitions():
    """Repeated calls move the GP mechanism across its initialization branches."""
    gp = mechanisms.GaussianProcessAdd_Mechanism(ncauses=1, points=8)
    causes = np.linspace(-1, 1, 8).reshape(8, 1)
    for _ in range(6):
        result = gp(causes)
        assert result.shape == (8, 1)


def test_gaussian_process_mix_with_and_without_causes():
    """GP mix should accept both multi-cause input and the noise-only fallback."""
    gp = mechanisms.GaussianProcessMix_Mechanism(ncauses=1, points=6)
    causes = np.linspace(-1, 1, 6).reshape(6, 1)
    first = gp(causes)
    assert first.shape == (6, 1)
    gp_empty = mechanisms.GaussianProcessMix_Mechanism(ncauses=0, points=5)
    assert gp_empty(np.empty((5, 0))).shape == (5, 1)


def test_gaussian_cause_and_noise_helpers():
    assert mechanisms.gaussian_cause(4).shape == (4,)
    assert mechanisms.noise(4, v=0.5).shape == (4, 1)


def test_compute_gauss_kernel_identity():
    x = np.zeros((3, 1))
    kernel = mechanisms.computeGaussKernel(x)
    assert np.allclose(np.diag(kernel), 1.0)
