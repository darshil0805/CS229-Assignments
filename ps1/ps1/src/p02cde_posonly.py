import numpy as np
import util

from p01b_logreg import LogisticRegression

# Character to replace with sub-problem letter in plot_path/pred_path
WILDCARD = 'X'


def main(train_path, valid_path, test_path, pred_path):
    """Problem 2: Logistic regression for incomplete, positive-only labels.

    Run under the following conditions:
        1. on y-labels,
        2. on l-labels,
        3. on l-labels with correction factor alpha.

    Args:
        train_path: Path to CSV file containing training set.
        valid_path: Path to CSV file containing validation set.
        test_path: Path to CSV file containing test set.
        pred_path: Path to save predictions.
    """
    pred_path_c = pred_path.replace(WILDCARD, 'c')
    pred_path_d = pred_path.replace(WILDCARD, 'd')
    pred_path_e = pred_path.replace(WILDCARD, 'e')

    # *** START CODE HERE ***
    x_train,y_train = util.load_dataset(train_path,label_col = 't',add_intercept = True)
    x_test,y_test = util.load_dataset(test_path,label_col = 't',add_intercept = True)
    model_c = LogisticRegression()
    model_c.fit(x_train,y_train)
    np.savetxt(pred_path_c,model_c.predict(x_test))
    util.plot(x_test,y_test,model_c.theta,'output/p02c.png')

    model_d = LogisticRegression()
    x_train,y_train = util.load_dataset(train_path,label_col = 'y',add_intercept = True)
    x_test,y_test = util.load_dataset(test_path,label_col = 't',add_intercept = True)
    model_d.fit(x_train,y_train)
    np.savetxt(pred_path_c,model_d.predict(x_test))
    util.plot(x_test,y_test,model_d.theta,'output/p02d.png')

    x_valid,y_valid = util.load_dataset(valid_path,label_col = 't',add_intercept = True)

    x_valid_e = [x_valid[i] for i in range(len(x_valid)) if y_valid[i] == 1]
    x_pred_sum = (model_d.predict(x_valid)).sum()
    alpha = x_pred_sum/len(x_valid)
    util.plot(x_test,y_test,model_d.theta,'output/p02e.png',alpha)

    # Part (c): Train and test on true labels
    # Make sure to save outputs to pred_path_c
    # Part (d): Train on y-labels and test on true labels
    # Make sure to save outputs to pred_path_d
    # Part (e): Apply correction factor using validation set and test on true labels
    # Plot and use np.savetxt to save outputs to pred_path_e
    # *** END CODER HERE
