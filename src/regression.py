"""
src/regression.py

Task 4 - Least-squares linear regression implemented by hand with NumPy.

Implements:
    4.3 hypothesis(X, theta)
    4.3 cost(X, y, theta)
    4.4 fit_normal(X, y)
    4.5 fit_batch_gd(X, y, alpha, n_iters)
    4.6 fit_sgd(X, y, alpha, n_epochs)
    3.1 rmse(y_true, y_pred)

No scikit-learn, statsmodels, or scipy are used.
"""

import numpy as np


# ---------------------------------------------------------------------
# 4.3 Hypothesis
# ---------------------------------------------------------------------

def hypothesis(X, theta):
    """
    Compute predictions for all rows.

    Mathematical form:
        h_theta(x) = theta^T x

    For the complete design matrix:
        h = X theta

    Parameters
    ----------
    X : numpy.ndarray
        Design matrix of shape (n, d+1).
        First column must be the intercept column of ones.

    theta : numpy.ndarray
        Parameter vector of shape (d+1,).

    Returns
    -------
    numpy.ndarray
        Predictions of shape (n,).
    """
    X = np.asarray(X, dtype=float)
    theta = np.asarray(theta, dtype=float)

    return X @ theta


# ---------------------------------------------------------------------
# 4.3 Cost
# ---------------------------------------------------------------------

def cost(X, y, theta):
    """
    Least-squares cost.

    J(theta) = 1/2 * sum_i (h_theta(x_i) - y_i)^2

    This is the cost formulation used with the gradient-descent
    update specified in the assignment.

    Parameters
    ----------
    X : numpy.ndarray
        Design matrix.

    y : numpy.ndarray
        Target values.

    theta : numpy.ndarray
        Model parameters.

    Returns
    -------
    float
        Cost J(theta).
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    theta = np.asarray(theta, dtype=float)

    predictions = hypothesis(X, theta)
    errors = predictions - y

    return float(0.5 * np.sum(errors ** 2))


# ---------------------------------------------------------------------
# 4.4 Normal Equation
# ---------------------------------------------------------------------

def fit_normal(X, y):
    """
    Fit linear regression using the normal equation.

    theta = (X^T X)^(-1) X^T y

    We use np.linalg.solve instead of explicitly calculating
    the inverse because it is numerically more stable.

    If X^T X is singular, np.linalg.lstsq is used as a fallback.

    Parameters
    ----------
    X : numpy.ndarray
        Training design matrix.

    y : numpy.ndarray
        Training targets.

    Returns
    -------
    numpy.ndarray
        Learned theta vector.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)

    XtX = X.T @ X
    Xty = X.T @ y

    try:
        theta = np.linalg.solve(XtX, Xty)
    except np.linalg.LinAlgError:
        print("Warning: X^T X is singular. Using least-squares fallback.")
        theta, *_ = np.linalg.lstsq(X, y, rcond=None)

    return theta


# ---------------------------------------------------------------------
# 4.5 Batch Gradient Descent
# ---------------------------------------------------------------------

def fit_batch_gd(X, y, alpha, n_iters):
    """
    Batch gradient descent.

    Starts from:
        theta = 0

    Update from the assignment:

        theta_j :=
            theta_j
            + alpha * sum_i(
                (y_i - h_theta(x_i)) * x_ij
              )

    The cost is recorded AFTER every update.

    Parameters
    ----------
    X : numpy.ndarray
        Training design matrix.

    y : numpy.ndarray
        Training targets.

    alpha : float
        Learning rate.

    n_iters : int
        Number of gradient-descent iterations.

    Returns
    -------
    theta : numpy.ndarray
        Final parameter vector.

    cost_history : numpy.ndarray
        Cost after each iteration.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)

    n, num_features = X.shape

    theta = np.zeros(num_features, dtype=float)

    cost_history = np.zeros(n_iters, dtype=float)

    for iteration in range(n_iters):

        # Current predictions
        predictions = hypothesis(X, theta)

        # Error:
        # y_i - h_theta(x_i)
        errors = y - predictions

        # Sum over all training examples.
        #
        # X.T @ errors gives:
        #
        # sum_i (y_i - h_i) * x_ij
        #
        gradient_sum = X.T @ errors

        # Assignment's update rule
        theta = theta + alpha * gradient_sum

        # Record J(theta) after the update
        cost_history[iteration] = cost(X, y, theta)

    return theta, cost_history


# ---------------------------------------------------------------------
# 4.6 Stochastic Gradient Descent
# ---------------------------------------------------------------------

def fit_sgd(X, y, alpha, n_epochs, random_state=0):
    """
    Stochastic gradient descent.

    Starts from:
        theta = 0

    In each epoch, training rows are processed one at a time.

    Assignment update:

        theta_j :=
            theta_j
            + alpha * (
                y_i - h_theta(x_i)
              ) * x_ij

    The assignment says not to shuffle the time series, therefore
    rows are processed in their existing chronological order.

    Cost is recorded after every complete epoch.

    Parameters
    ----------
    X : numpy.ndarray
        Training design matrix.

    y : numpy.ndarray
        Training targets.

    alpha : float
        Learning rate.

    n_epochs : int
        Number of epochs.

    random_state : int
        Kept for API compatibility/reproducibility. No shuffling is
        performed because the assignment explicitly requires
        chronological ordering.

    Returns
    -------
    theta : numpy.ndarray
        Final parameter vector.

    cost_history : numpy.ndarray
        Full training cost after every epoch.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)

    n, num_features = X.shape

    theta = np.zeros(num_features, dtype=float)

    cost_history = np.zeros(n_epochs, dtype=float)

    for epoch in range(n_epochs):

        # IMPORTANT:
        # Do NOT shuffle.
        # The assignment uses chronological training data.
        for i in range(n):

            xi = X[i]
            yi = y[i]

            # Prediction for one row
            prediction = float(xi @ theta)

            # Error
            error = yi - prediction

            # SGD update
            theta = theta + alpha * error * xi

        # Record complete training-set cost
        cost_history[epoch] = cost(X, y, theta)

    return theta, cost_history


# ---------------------------------------------------------------------
# RMSE
# ---------------------------------------------------------------------

def rmse(y_true, y_pred):
    """
    Root Mean Square Error.

    RMSE =
        sqrt(
            1/m * sum_i (y_i - y_hat_i)^2
        )

    This returns the RMSE in the same units as y.

    For this assignment, that means kW.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    if y_true.size == 0:
        raise ValueError("Cannot calculate RMSE: no observations provided.")

    return float(
        np.sqrt(
            np.mean((y_true - y_pred) ** 2)
        )
    )


# ---------------------------------------------------------------------
# Simple smoke test
# ---------------------------------------------------------------------

if __name__ == "__main__":

    # Synthetic relationship:
    #
    # y = 2 + 3x

    rng = np.random.default_rng(0)

    x = rng.uniform(-5, 5, size=200)

    y = 2 + 3 * x

    X = np.column_stack([
        np.ones_like(x),
        x
    ])

    print("Running regression.py smoke test...\n")

    # Normal equation
    theta_normal = fit_normal(X, y)

    # Batch GD
    theta_bgd, hist_bgd = fit_batch_gd(
        X,
        y,
        alpha=0.001,
        n_iters=2000
    )

    # SGD
    theta_sgd, hist_sgd = fit_sgd(
        X,
        y,
        alpha=0.001,
        n_epochs=50
    )

    print("True theta:")
    print("[2, 3]\n")

    print("Normal equation:")
    print(theta_normal)
    print()

    print("Batch GD:")
    print(theta_bgd)
    print()

    print("SGD:")
    print(theta_sgd)
    print()

    print(
        "Batch GD cost decreases:",
        np.all(np.diff(hist_bgd) <= 1e-6)
    )

    predictions = hypothesis(X, theta_normal)

    print(
        "Normal equation RMSE:",
        rmse(y, predictions)
    )