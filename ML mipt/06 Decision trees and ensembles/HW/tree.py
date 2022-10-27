import numpy as np
from sklearn.base import BaseEstimator

tree_depth = 0

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
    class_prob_array = np.array([y[:, i].sum() / y.shape[0] for i in range(num_classes)])
    return -(class_prob_array * np.log(class_prob_array + EPS)).sum()


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
    class_prob_array = np.array([y[:, i].sum() / y.shape[0] for i in range(num_classes)])
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
    return (((y - y.mean())**2).sum()/y.shape[0])


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
    return (np.abs(y - np.median(y))).sum()/y.shape[0]


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
        #print('split:', x_left.shape, x_right.shape, x_subset.shape)
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
        min_h = np.inf  # entropy or gini
        for feature, threshold_list in potential_splits.items():
            for threshold in threshold_list:
                left_y, right_y = self.make_split_only_y(feature, threshold, x_subset, y_subset)
                temp_h = 999
                if (left_y.shape[0] > self.min_samples_split) and (right_y.shape[0] > self.min_samples_split):
                    temp_h = (left_y.shape[0] / y_subset.shape[0]) * self.criterion(left_y) + \
                             (right_y.shape[0] / y_subset.shape[0]) * self.criterion(right_y)
                    #print(temp_h)
                    #print(left_y.shape[0], right_y.shape[0], right_y.shape[0] + left_y.shape[0])
                #print(temp_h)
                    if temp_h < min_h:
                        min_h = temp_h
                        best_threshold = threshold
                        best_feature = feature
        #print("min_h = ", min_h)
        #print('Bf ', best_feature, " BT ", best_threshold)
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

    def make_tree(self, x_subset, y_subset, current_depth=0):
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
        new_node = None
        #print(current_depth)
        if (x_subset.shape != 1):
            if (current_depth!=self.max_depth):
                best_feature, best_threshold = self.choose_best_split(x_subset, y_subset)
                if (best_feature is not None) and (best_threshold is not None):
                    (x_left, y_left), (x_right, y_right) = self.make_split(best_feature, best_threshold, x_subset, y_subset)
                    new_node = Node(best_feature, best_threshold)
                    #print(y_left.shape, y_right.shape)
                    if self.classification:
                        new_node.proba = np.sum(y_subset, axis=0)/y_subset.shape[0]
                    else:
                        new_node.proba = np.mean(y_subset)
                    #print(new_node.proba.shape)
                    new_node.left_child = self.make_tree(x_left, y_left, current_depth+1)
                    new_node.right_child = self.make_tree(x_right, y_right, current_depth+1)
        return new_node

    def predict(self, X):
        """
            Predict the target value or class label  the model from scratch using the provided data

            Parameters
            ----------
            X : np.array of type float with shape (n_objects, n_features)
                Feature matrix representing the data the predictions should be provided for
            Returns
            -------
            y_predicted : np.array of type int with shape (n_objects, 1) in classification
                       (n_objects, 1) in regression
                Column vector of class labels in classification or target values in regression

        """
        y_predicted = np.zeros((X.shape[0], 1))
        if self.classification:
            x_temp = X
        for i, sample in zip(range(X.shape[0]), X):
            tree = self.root
            while (tree.left_child is not None) and (tree.right_child is not None):
                if (sample[tree.feature_index] < tree.value) and (tree.left_child is not None):
                    #if self.classification:
                    #    X = X[tree.feature_index] > tree.
                    tree = tree.left_child
                elif (tree.right_child is not None):
                    tree = tree.right_child
            #print(tree.proba, np.argmax(tree.proba))
            if self.classification:
                y_predicted[i] = np.argmax(tree.proba)
            else:
                y_predicted[i] = tree.proba
        return y_predicted

    def predict_proba(self, X):
        """
                Only for classification
                Predict the class probabilities using the provided data

                Parameters
                ----------
                X : np.array of type float with shape (n_objects, n_features)
                    Feature matrix representing the data the predictions should be provided for
                Returns
                -------
                y_predicted_probs : np.array of type float with shape (n_objects, n_classes)
                    Probabilities of each class for the provided objects

        """
        assert self.classification, 'Available only for classification problem'
        y_predicted = np.zeros((X.shape[0], self.n_classes))
        for i, sample in zip(range(X.shape[0]), X):
            tree = self.root
            while (tree.left_child is not None) and (tree.right_child is not None):
                if (sample[tree.feature_index] < tree.value) and (tree.left_child is not None):
                    tree = tree.left_child
                elif (tree.right_child is not None):
                    tree = tree.right_child
            #print(tree.proba, np.argmax(tree.proba))
            y_predicted[i, :] = tree.proba
        return y_predicted