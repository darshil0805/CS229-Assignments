import matplotlib.pyplot as plt
import numpy as np
import util

from linear_model import LinearModel


def main(tau, train_path, eval_path):
    """Problem 5(b): Locally weighted regression (LWR)

    Args:
        tau: Bandwidth parameter for LWR.
        train_path: Path to CSV file containing dataset for training.
        eval_path: Path to CSV file containing dataset for evaluation.
    """
    # Load training set
    x_train, y_train = util.load_dataset(train_path, add_intercept=True)
    model = LocallyWeightedLinearRegression(tau = tau)
    model.fit(x_train,y_train)

    x_eval, y_eval = util.load_dataset(eval_path, add_intercept=True)
    y_pred = model.predict(x_eval)

    mse = np.mean((y_eval - y_pred)**2)
    print(f'mse is {mse}')

    plt.figure(figsize = (16,8))
    plt.plot(x_train, y_train, 'bx', linewidth=2)
    plt.plot(x_eval, y_pred, 'ro', linewidth=2)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.savefig('output/p05b.png')
    # *** START CODE HERE ***
    # Fit a LWR model
    # Get MSE value on the validation set
    # Plot validation predictions on top of training set
    # No need to save predictions
    # Plot data
    # *** END CODE HERE ***


class LocallyWeightedLinearRegression(LinearModel):
    """Locally Weighted Regression (LWR).

    Example usage:
        > clf = LocallyWeightedLinearRegression(tau)
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """

    def __init__(self, tau):
        super(LocallyWeightedLinearRegression, self).__init__()
        self.tau = tau
        self.x = None
        self.y = None

    def fit(self, x, y):
        """Fit LWR by saving the training set.

        """
        # *** START CODE HERE ***
        self.x = x
        self.y = y



        # *** END CODE HERE ***

    def predict(self, x):
        """Make predictions given inputs x.


        Args:
            x: Inputs of shape (m, n).

        Returns:
            Outputs of shape (m,).
        """
        # *** START CODE HERE ***
        m,n = np.shape(x)
        s = (self.x).shape[0]
        y = np.zeros(m)
        W = np.zeros((s,s))

        for i in range(m):

            for j in range(s):
                a = np.exp(-((np.linalg.norm(self.x[j]-x[i]))**2) / (2 * self.tau**2))
                W[j,j] = a
            a = self.x.T.dot(W).dot(self.x)
            a_inv = np.linalg.inv(a)
            b = self.x.T.dot(W).dot(self.y)
            self.theta =  a_inv.dot(b)
            y[i] = (self.theta).T.dot(x[i])

        return y
        # *** END CODE HERE ***
