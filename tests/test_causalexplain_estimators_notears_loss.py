import numpy as np

from causalexplain.estimators.notears import loss


def test_least_squares_loss_and_grad():
    data = np.array([[1.0, 2.0], [3.0, 4.0]])
    W = np.array([[0.0, 0.5], [0.0, 0.0]])
    cov = np.cov(data.T)
    d, n = data.shape[1], data.shape[0]

    computed_loss = loss.least_squares_loss(W, data, cov, d, n)
    expected_loss = 0.5 / n * np.linalg.norm(data - data @ W, ord="fro") ** 2
    assert np.isclose(computed_loss, expected_loss)

    grad = loss.least_squares_loss_grad(W, data, cov, d, n)
    expected_grad = (-1.0 / n) * data.T @ (data - data @ W)
    assert np.allclose(grad, expected_grad)


def test_least_squares_cov_variants():
    data = np.array([[1.0, 0.0]])
    W = np.array([[0.0]])
    cov = np.array([[2.0]])
    d, n = 1, 1

    loss_cov = loss.least_squares_loss_cov(W, data, cov, d, n)
    expected_loss = 0.5 * np.trace((np.eye(d) - W).T @ cov.T @ (np.eye(d) - W))
    assert np.isclose(loss_cov, expected_loss)

    grad_cov = loss.least_squares_loss_cov_grad(W, data, cov, d, n)
    expected_grad = (-0.5) * ((cov + cov.T) @ (np.eye(d) - W))
    assert np.allclose(grad_cov, expected_grad)
