import matplotlib.pyplot as plt
import numpy as np
import util

from p05b_lwr import LocallyWeightedLinearRegression


def main(tau_values, train_path, valid_path, test_path, pred_path):
    """Problem 5(b): Tune the bandwidth paramater tau for LWR.

    Args:
        tau_values: List of tau values to try.
        train_path: Path to CSV file containing training set.
        valid_path: Path to CSV file containing validation set.
        test_path: Path to CSV file containing test set.
        pred_path: Path to save predictions.
    """
    # Load training set
    x_train, y_train = util.load_dataset(train_path, add_intercept=True)
    x_valid, y_valid = util.load_dataset(valid_path, add_intercept=True)
    x_test, y_test = util.load_dataset(test_path, add_intercept=True)
    mse = 1
    best_tau = 0
    for tau in tau_values:
        model = LocallyWeightedLinearRegression(tau = tau)
        model.fit(x_train,y_train)
        y_pred = model.predict(x_test)

        ms = np.mean((y_pred-y_test)**2)

        plt.figure(figsize = (16,8))
        plt.plot(x_train,y_train,'bx',linewidth = 2)
        plt.plot(x_test,y_pred,'ro',linewidth =2)
        plt.xlabel('x')
        plt.ylabel('y')
        plt.savefig(f'output/p05c{tau}.png')
        print(f'The mse for tau = {tau} is {ms}')
        if ms < mse:
            mse = ms
            best_tau = tau
    model = LocallyWeightedLinearRegression(tau = best_tau)
    model.fit(x_train,y_train)
    y_pred = model.predict(x_test)
    np.savetxt(pred_path,y_pred)
    mse = np.mean((y_pred - y_test)**2)
    print(f'test set: tau={best_tau}, MSE={mse}')

    # *** START CODE HERE ***
    # Search tau_values for the best tau (lowest MSE on the validation set)
    # Fit a LWR model with the best tau value
    # Run on the test set to get the MSE value
    # Save predictions to pred_path
    # Plot data
    # *** END CODE HERE ***
