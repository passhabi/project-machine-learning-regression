import numpy as np
from learnt.regression import predict_outcome


def simple_linear_regression(input_feature, output):
    """
    :param input_feature: vector (numpy array) feature sqft_living
    :param output: vector of price
    :return: W0 , W1
    """
    n = output.size
    sum_y = output.sum()
    sum_x = input_feature.sum()
    sum_xy = (output * input_feature).sum()
    sum_x_squared = (input_feature ** 2).sum()

    numerator = sum_xy - (sum_y * sum_x / n)
    denominator = sum_x_squared - (sum_x * sum_x / n)
    slope = numerator / denominator
    intercept = (sum_y / n) - slope * (sum_x / n)
    return intercept, slope


def regression_gradient_descent(feature_matrix, output, initial_weights, step_size, tolerance):
    """
    **Gradient descent Algorithm**


    :param feature_matrix:
    :param output:
    :param initial_weights: 1×n numpy array
    :param step_size:
    :param tolerance:
    :return:
    """
    converged = False
    weights = np.array(initial_weights).reshape(-1, 1)
    while not converged:
        # compute the predictions based on feature_matrix and weights:
        predictions = predict_outcome(feature_matrix, weights)
        # compute the errors as predictions - output:
        errors = predictions - output
        gradient_sum_squares = 0  # initialize the gradient
        # while not converged, update each weight individually:
        for i in range(len(weights)):
            # Recall that feature_matrix[:, i] is the feature column associated with weights[i]
            # compute the derivative for weight[i]:
            derivative = feature_derivative(errors, feature_matrix[:, i])
            # add the squared derivative to the gradient magnitude
            gradient_sum_squares = + derivative ** 2
            # update the weight based on step size and derivative:
            weights[i] = weights[i] + step_size * derivative
        gradient_magnitude = np.sqrt(gradient_sum_squares)
        if gradient_magnitude < tolerance:
            converged = True
    return weights


def feature_derivative(errors, feature):
    # -2H^T(y-HW)   =>  -2H^T.(error)    =>  2 * H^T . error
    feature = 2 * feature
    derivative = np.dot(errors, feature)
    return derivative
