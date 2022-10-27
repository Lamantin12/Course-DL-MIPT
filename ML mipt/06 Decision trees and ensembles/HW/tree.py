import numpy as np
from sklearn.metrics import BaseEstimator


def entropy(y):
    """
        Computes entropy of the provided distribution. Use log(value + eps) for numerical stability

        Parameters
        ----------
        y : np.array of type float with shape (n_objects, n_classes)
            One-hot representation of class labels for corresponding subset

        Returns
        -------
        float
            Entropy of the provided subset
    """

    EPS = 0.0005
    num_classes = y.shape[1]
    class_prob_array = np.array([y[:, i].sum() / y_shape[0] for i in range(num_classes)])
    return (class_prob_array * np.log(class_prob_array + EPS)).sum()


def gini(y):
    """
        Computes the Gini impurity of the provided distribution

        Parameters
        ----------
        y : np.array of type float with shape (n_objects, n_classes)
            One-hot representation of class labels for corresponding subset

        Returns
        -------
        float
            Gini impurity of the provided subset
    """
    num_classes = y.shape[1]
    class_prob_array = np.array([y[:, i].sum() / y_shape[0] for i in range(num_classes)])
    return 1 - (class_prob_array**2).sum()


def variance(y):
    """
        Computes the variance the provided target values subset

        Parameters
        ----------
        y : np.array of type float with shape (n_objects, 1)
            Target values vector

        Returns
        -------
        float
            Variance of the provided target vector
    """
    return ((y - y.mean())**2).sum()/y.shape[0]


def mad_median(y):
    """
        Computes the mean absolute deviation from the median in the
        provided target values subset

        Parameters
        ----------
        y : np.array of type float with shape (n_objects, 1)
            Target values vector

        Returns
        -------
        float
            Mean absolute deviation from the median in the provided vector
    """
    return (np.abs(y - y.median())).sum()/y.shape[0]


def one_hot_encode(n_classes, y):
    y_one_hot = np.zeros((len(y), n_classes), dtype=float)
    y_one_hot[np.arange(len(y)), y.astype(int)[:, 0]] = 1
    return y_one_hot


def one_hot_decode(y_one_hot):
    return y_one_hot.argmax(axis=1)[:, None]


class Node:
    def __init__(self, feature_index, threshold, proba=0):
        self.feature_index = feature_index
        self.value = threshold
        self.proba = proba
        self.left_child = None
        self.right_child = None


class DecisionTree(BaseEstimator):
    all_criterions = {
        'gini': (gini, True), # (criterion, classification flag)
        'entropy': (entropy, True),
        'variance': (variance, False),
        'mad_median': (mad_median, False)
    }

    def __init__(self, n_classes=None, max_depth=np.inf, min_samples_split=2,
                 criterion_name='gini', debug=False):

        assert criterion_name in self.all_criterions.keys(), f'Criterion name must be the following {self.all_criterions.keys()}'

        self.n_classes = n_classes
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.criterion_name = criterion_name

        self.depth = 0
        self.root = None
        self.debug = debug

    def make_split(self, feature_index, threshold, x_subset, y_subset):
        """
                Makes split of the provided data subset and target values using provided feature and threshold

                Parameters
                ----------
                feature_index : int
                    Index of feature to make split with
                threshold : float
                    Threshold value to perform split
                x_subset : np.array of type float with shape (n_objects, n_features)
                    Feature matrix representing the selected subset
                y_subset : np.array of type float with shape (n_objects, n_classes) in classification
                           (n_objects, 1) in regression
                    One-hot representation of class labels for corresponding subset

                Returns
                -------
                (X_left, y_left) : tuple of np.arrays of same type as input X_subset and y_subset
                    Part of the providev subset where selected feature x^j < threshold
                (X_right, y_right) : tuple of np.arrays of same type as input X_subset and y_subset
                    Part of the providev subset where selected feature x^j >= threshold
            """
        left = x_subset[:, feature_index] < threshold
        x_left = x_subset[left]
        y_left = y_subset[left]
        right = x_subset[:, feature_index] >= threshold
        x_right = x_subset[right]
        y_right = y_subset[right]

        return (x_left, y_left), (x_right, y_right)

    def make_split_only_y(self, feature_index, threshold, x_subset, y_subset):
        """
            Split only target values into two subsets with specified feature and threshold

            Parameters
            ----------
            feature_index : int
                Index of feature to make split with
            threshold : float
                Threshold value to perform split
            x_subset : np.array of type float with shape (n_objects, n_features)
                Feature matrix representing the selected subset
            y_subset : np.array of type float with shape (n_objects, n_classes) in classification
                       (n_objects, 1) in regression
                One-hot representation of class labels for corresponding subset
            Returns
            -------
            y_left : np.array of type float with shape (n_objects_left, n_classes) in classification
                       (n_objects, 1) in regression
                Part of the provided subset where selected feature x^j < threshold
            y_right : np.array of type float with shape (n_objects_right, n_classes) in classification
                       (n_objects, 1) in regression
                Part of the provided subset where selected feature x^j >= threshold
        """
        left = x_subset[:, feature_index] < threshold
        y_left = y_subset[left]
        right = x_subset[:, feature_index] >= threshold
        y_right = y_subset[right]

        return y_left, y_right


    def choose_best_split(self, x_subset, y_subset):
        """
                Greedily select the best feature and best threshold w.r.t. selected criterion

                Parameters
                ----------
                x_subset : np.array of type float with shape (n_objects, n_features)
                    Feature matrix representing the selected subset
                y_subset : np.array of type float with shape (n_objects, n_classes) in classification
                           (n_objects, 1) in regression
                    One-hot representation of class labels or target values for corresponding subset

                Returns
                -------
                feature_index : int
                    Index of feature to make split with
                threshold : float
                    Threshold value to perform split
        """
        potential_splits = {}
        for column in range(x_subset.shape[1]):
            unique_column_x_subset = np.unique(x_subset[:, column])
            potential_splits[column] = unique_column_x_subset

        best_feature = None
        best_threshold = None
        max_h = np.inf  # entropy or gini
        for feature, threshold_list in potential_splits:
            for threshold in threshold_list:
                left_y, right_y = self.make_split_only_y(feature, threshold, x_subset, y_subset)
                temp_h = (left_y.shape[0]/y_subset.shape[0])*self.criterion(left_y) + \
                         (right_y.shape[0] / y_subset.shape[0]) * self.criterion(right_y)
                if temp_h < max_h:
                    best_threshold = threshold
                    best_feature = feature

        return best_feature, best_threshold

    def fit(self, X, y):
        """
                Fit the model from scratch using the provided data

                Parameters
                ----------
                X : np.array of type float with shape (n_objects, n_features)
                   Feature matrix representing the data to train on
                y : np.array of type int with shape (n_objects, 1) in classification
                           of type float with shape (n_objects, 1) in regression
                   Column vector of class labels in classification or target values in regression

        """
        assert len(y.shape) == 2 and len(y) == len(X), 'Wrong y shape'
        self.criterion, self.classification = self.all_criterions[self.criterion_name]
        if self.classification:
            if self.n_classes is None:
                self.n_classes = len(np.unique(y))
            y = one_hot_encode(self.n_classes, y)

        self.root = self.make_tree(X, y)

    def make_tree(self, x_subset, y_subset):
        """
            Recursively builds the tree

            Parameters
            ----------
            x_subset : np.array of type float with shape (n_objects, n_features)
                Feature matrix representing the selected subset
            y_subset : np.array of type float with shape (n_objects, n_classes) in classification
                (n_objects, 1) in regression
                One-hot representation of class labels or target values for corresponding subset

            Returns
            -------
            root_node : Node class instance
                Node of the root of the fitted tree
        """
        
        return new_node