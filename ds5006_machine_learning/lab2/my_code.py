import numpy as np

def add_intercept(X):

    ones = np.ones((X.shape[0], 1))
    print("Shape of ones:", ones.shape)
    print(ones)
    print("Shape of X:", X.shape)
    return np.hstack((X, ones))

def one_hot(labels, n_classes):
    result = np.zeros((labels.size, n_classes), dtype=float)
    result[np.arange(labels.size), labels] = 1.0
    return result

def sigmoid(values):
    values = np.asarray(values, dtype=float)
    result = np.empty_like(values)

    positive = values >= 0
    result[positive] = 1.0 / (1.0 + np.exp(-values[positive]))

    exp_values = np.exp(values[~positive])
    result[~positive] = exp_values / (1.0 + exp_values)
    return result

def softmax_rows(scores):
    scores = np.asarray(scores, dtype=float)

    shifted = scores - np.max(scores, axis=1, keepdims=True)
    print("Shifted scores:\n", shifted)
    exp_scores = np.exp(shifted)

    return exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

def quadratic_expansion(design):
    a = design[:, 0]
    b = design[:, 1]
    return np.column_stack((
        a,
        b,
        a**2,
        a*b,
        b**2
    ))

def fit_indicator_regression(design, targets):
    q_matrix, r_matrix = np.linalg.qr(design, mode='reduced')
    return np.linalg.solve(r_matrix, q_matrix.T @ targets)

def indicator_predict(design, coefficients):
    return np.argmax(design @ coefficients, axis=1)

def main():
    # Test the add_intercept function
    # X = np.array([[1, 2],
    #              [3, 4],
    #              [5, 6]])
    # X_with_intercept = add_intercept(X)
    # print("Original X:\n", X)
    # print("X with intercept:\n", X_with_intercept)
    # labels = np.array([0, 1, 2, 1])
    # print("One-hot encoding of labels:\n", one_hot(labels, 3))
    # print("Sigmoid of values:\n", sigmoid(np.array([-1500, 0, 1500])))
    # print("Softmax of scores:\n", softmax_rows(np.array([[10000, 9998, 9999],
    #                                                     [1, 2, 3],
    #                                                     [750, 752, 751]])))
    design = np.array([
        [1, 2, 1],
        [3, 4, 1],
        [5, 6, 1]
        ])
    # print("Quadratic expansion of design:\n", quadratic_expansion(design))
    targets = np.array([
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1]
        ])
    coefficients = fit_indicator_regression(design, targets)
    predictions = indicator_predict(design, coefficients)
    print("Fitted coefficients:\n", coefficients)
    print("Predictions:\n", predictions)

if __name__ == "__main__":
    main()
