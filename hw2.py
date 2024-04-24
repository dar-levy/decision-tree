import queue

import numpy as np
import matplotlib.pyplot as plt

### Chi square table values ###
# The first key is the degree of freedom 
# The second key is the p-value cut-off
# The values are the chi-statistic that you need to use in the pruning

chi_table = {1: {0.5 : 0.45,
             0.25 : 1.32,
             0.1 : 2.71,
             0.05 : 3.84,
             0.0001 : 100000},
         2: {0.5 : 1.39,
             0.25 : 2.77,
             0.1 : 4.60,
             0.05 : 5.99,
             0.0001 : 100000},
         3: {0.5 : 2.37,
             0.25 : 4.11,
             0.1 : 6.25,
             0.05 : 7.82,
             0.0001 : 100000},
         4: {0.5 : 3.36,
             0.25 : 5.38,
             0.1 : 7.78,
             0.05 : 9.49,
             0.0001 : 100000},
         5: {0.5 : 4.35,
             0.25 : 6.63,
             0.1 : 9.24,
             0.05 : 11.07,
             0.0001 : 100000},
         6: {0.5 : 5.35,
             0.25 : 7.84,
             0.1 : 10.64,
             0.05 : 12.59,
             0.0001 : 100000},
         7: {0.5 : 6.35,
             0.25 : 9.04,
             0.1 : 12.01,
             0.05 : 14.07,
             0.0001 : 100000},
         8: {0.5 : 7.34,
             0.25 : 10.22,
             0.1 : 13.36,
             0.05 : 15.51,
             0.0001 : 100000},
         9: {0.5 : 8.34,
             0.25 : 11.39,
             0.1 : 14.68,
             0.05 : 16.92,
             0.0001 : 100000},
         10: {0.5 : 9.34,
              0.25 : 12.55,
              0.1 : 15.99,
              0.05 : 18.31,
              0.0001 : 100000},
         11: {0.5 : 10.34,
              0.25 : 13.7,
              0.1 : 17.27,
              0.05 : 19.68,
              0.0001 : 100000}}

def calc_gini(data):
    """
    Calculate gini impurity measure of a dataset.
 
    Input:
    - data: any dataset where the last column holds the labels.
 
    Returns:
    - gini: The gini impurity value.
    """
    gini = 0.0
    ###########################################################################
    # TODO: Implement the function.                                           #
    ###########################################################################
    class_column = data[:, -1]
    classes, count = np.unique(class_column, return_counts=True)
    weight = count / len(data)
    gini = 1 - np.sum(weight ** 2)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return gini

def calc_entropy(data):
    """
    Calculate the entropy of a dataset.

    Input:
    - data: any dataset where the last column holds the labels.

    Returns:
    - entropy: The entropy value.
    """
    entropy = 0.0
    ###########################################################################
    # TODO: Implement the function.                                           #
    ###########################################################################
    class_column = data[:, -1]
    classes, count = np.unique(class_column, return_counts=True)
    weight = count / len(data)
    entropy = (-1) * np.sum(weight * np.log2(weight))
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return entropy

class DecisionNode:

    
    def __init__(self, data, impurity_func, feature=-1,depth=0, chi=1, max_depth=1000, gain_ratio=False):
        
        self.data = data # the relevant data for the node
        self.feature = feature # column index of criteria being tested
        self.pred = self.calc_node_pred() # the prediction of the node
        self.depth = depth # the current depth of the node
        self.children = [] # array that holds this nodes children
        self.children_values = []
        self.terminal = False # determines if the node is a leaf
        self.chi = chi 
        self.max_depth = max_depth # the maximum allowed depth of the tree
        self.impurity_func = impurity_func
        self.gain_ratio = gain_ratio
        self.feature_importance = 0
    
    def calc_node_pred(self):
        """
        Calculate the node prediction.

        Returns:
        - pred: the prediction of the node
        """
        pred = None
        ###########################################################################
        # TODO: Implement the function.                                           #
        ###########################################################################
        classes, count = np.unique(self.data[:, -1], return_counts=True)
        classes_count = dict(zip(classes, count))
        pred = max(classes_count, key=classes_count.get)
        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################
        return pred
        
    def add_child(self, node, val):
        """
        Adds a child node to self.children and updates self.children_values

        This function has no return value
        """
        ###########################################################################
        # TODO: Implement the function.                                           #
        ###########################################################################
        self.children.append(node)
        self.children_values.append(val)
        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################
        
    def calc_feature_importance(self, n_total_sample):
        """
        Calculate the selected feature importance.
        
        Input:
        - n_total_sample: the number of samples in the dataset.

        This function has no return value - it stores the feature importance in 
        self.feature_importance
        """
        ###########################################################################
        # TODO: Implement the function.                                           #
        ###########################################################################
        current_node_impurity = self.impurity_func(self.data)
        total_instances = len(self.data)
        weighted_impurity_sum = self._calculate_weighted_sum(n_total_sample)
        self.feature_importance = (total_instances / n_total_sample) * (current_node_impurity - weighted_impurity_sum)
        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################

    def _calculate_weighted_sum(self, n_total_sample):
        weighted_impurity_sum = 0
        for child in self.children:
            child_impurity = self.impurity_func(child.data)
            child_samples = len(child.data)
            weighted_impurity_sum += (child_samples / n_total_sample) * child_impurity

        return weighted_impurity_sum

    def goodness_of_split(self, feature):
        """
        Calculate the goodness of split of a dataset given a feature and impurity function.

        Input:
        - feature: the feature index the split is being evaluated according to.

        Returns:
        - goodness: the goodness of split
        - groups: a dictionary holding the data after splitting 
                  according to the feature values.
        """
        goodness = 0
        groups = {} # groups[feature_value] = data_subset
        ###########################################################################
        # TODO: Implement the function.                                           #
        ###########################################################################
        total_instances = len(self.data)
        parent_impurity = self.impurity_func(self.data)
        children_weighted_impurity = 0
        unique_values = np.unique(self.data[:, feature])
        for value in unique_values:
            child_instances = self.data[self.data[:, feature] == value]
            groups[value] = child_instances
            subset_impurity = self.impurity_func(child_instances)
            proportion = len(child_instances) / total_instances
            children_weighted_impurity += proportion * subset_impurity

        goodness = parent_impurity - children_weighted_impurity
        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################
        return goodness, groups

    def split(self):
        """
        Splits the current node according to the self.impurity_func. This function finds
        the best feature to split according to and create the corresponding children.
        This function should support pruning according to self.chi and self.max_depth.

        This function has no return value
        """
        ###########################################################################
        # TODO: Implement the function.                                           #
        ###########################################################################
        if self.terminal or self.depth >= self.max_depth:
            return

        best_goodness = -float('inf')
        best_feature = None
        best_groups = None

        # Evaluate each feature for splitting
        n_features = self.data.shape[1] - 1  # assuming the last column is the target label
        for feature in range(n_features):
            goodness, groups = self.goodness_of_split(feature)
            if goodness > best_goodness:
                best_goodness = goodness
                best_feature = feature
                best_groups = groups

        # If no good split found or if the chi value suggests stopping
        if best_feature is None or best_goodness <= 0:
            self.terminal = True
            return

        # Apply chi-square pruning if necessary
        if self.chi > 0:
            chi_stat = self._calculate_chi_square(best_groups)
            if chi_stat < chi_table.get(len(best_groups) - 1, {}).get(self.chi, float('inf')):
                self.terminal = True
                return

        # Create child nodes
        self.feature = best_feature
        for value, subset in best_groups.items():
            if len(subset) > 0:
                child_node = DecisionNode(subset, self.impurity_func, feature=best_feature, depth=self.depth + 1,
                                          chi=self.chi, max_depth=self.max_depth)
                self.add_child(child_node, value)

        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################

    def _calculate_chi_square(self, groups):
        # Get the overall frequencies of the classes in the original data
        total_samples = len(self.data)
        class_counts = np.unique(self.data[:, -1], return_counts=True)[1]
        expected_ratios = class_counts / total_samples

        chi_stat = 0
        for group in groups.values():
            # Calculate expected counts for each class in this group
            group_size = len(group)
            expected_counts = expected_ratios * group_size
            observed_counts = np.array([len(group[group[:, -1] == class_val]) for class_val in np.unique(self.data[:, -1])])

            # Calculate the chi-square statistic for this group
            with np.errstate(divide='ignore', invalid='ignore'):
                chi_contributions = (observed_counts - expected_counts) ** 2 / expected_counts
                chi_contributions = np.nan_to_num(chi_contributions)  # handle zero division
            chi_stat += np.sum(chi_contributions)

        return chi_stat


class DecisionTree:
    def __init__(self, data, impurity_func, feature=-1, chi=1, max_depth=1000, gain_ratio=False):
        self.data = data # the relevant data for the tree
        self.impurity_func = impurity_func # the impurity function to be used in the tree
        self.chi = chi 
        self.max_depth = max_depth # the maximum allowed depth of the tree
        self.gain_ratio = gain_ratio #
        self.root = None # the root node of the tree
        
    def build_tree(self):
        """
        Build a tree using the given impurity measure and training dataset. 
        You are required to fully grow the tree until all leaves are pure 
        or the goodness of split is 0.

        This function has no return value
        """
        self.root = None
        ###########################################################################
        # TODO: Implement the function.                                           #
        ###########################################################################
        self.root = DecisionNode(data=self.data, impurity_func=self.impurity_func, depth=0, chi=self.chi, max_depth=self.max_depth, gain_ratio=self.gain_ratio)
        q = queue.Queue
        q.put(self.root)
        while not q.empty():
            current_node = q.get()
            if len(np.unique(current_node.data)) == 1:
                current_node.terminal = True
                continue

            current_node.split()
            for child in current_node.children:
                q.put(child)

        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################

    def predict(self, instance):
        """
        Predict a given instance
     
        Input:
        - instance: an row vector from the dataset. Note that the last element 
                    of this vector is the label of the instance.
     
        Output: the prediction of the instance.
        """
        pred = None
        ###########################################################################
        # TODO: Implement the function.                                           #
        ###########################################################################
        node = self.root
        while not node.terminal:
            feature = node.feature
            value = instance[feature]
            # Find the child node that corresponds to the feature value of the instance
            is_child_found = False
            for i, val in enumerate(node.children_values):
                if val == value:
                    node = node.children[i]
                    is_child_found = True
                    break
            if not is_child_found:
                break
        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################
        return node.pred

    def calc_accuracy(self, dataset):
        """
        Predict a given dataset 
     
        Input:
        - dataset: the dataset on which the accuracy is evaluated
     
        Output: the accuracy of the decision tree on the given dataset (%).
        """
        accuracy = 0
        ###########################################################################
        # TODO: Implement the function.                                           #
        ###########################################################################
        pass
        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################
        return accuracy
        
    def depth(self):
        return self.root.depth()

def depth_pruning(X_train, X_validation):
    """
    Calculate the training and validation accuracies for different depths
    using the best impurity function and the gain_ratio flag you got
    previously. On a single plot, draw the training and testing accuracy 
    as a function of the max_depth. 

    Input:
    - X_train: the training data where the last column holds the labels
    - X_validation: the validation data where the last column holds the labels
 
    Output: the training and validation accuracies per max depth
    """
    training = []
    validation  = []
    root = None
    for max_depth in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
        ###########################################################################
        # TODO: Implement the function.                                           #
        ###########################################################################
        pass
        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################
    return training, validation


def chi_pruning(X_train, X_test):

    """
    Calculate the training and validation accuracies for different chi values
    using the best impurity function and the gain_ratio flag you got
    previously. 

    Input:
    - X_train: the training data where the last column holds the labels
    - X_validation: the validation data where the last column holds the labels
 
    Output:
    - chi_training_acc: the training accuracy per chi value
    - chi_validation_acc: the validation accuracy per chi value
    - depth: the tree depth for each chi value
    """
    chi_training_acc = []
    chi_validation_acc  = []
    depth = []

    ###########################################################################
    # TODO: Implement the function.                                           #
    ###########################################################################
    pass
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
        
    return chi_training_acc, chi_testing_acc, depth


def count_nodes(node):
    """
    Count the number of node in a given tree
 
    Input:
    - node: a node in the decision tree.
 
    Output: the number of node in the tree.
    """
    ###########################################################################
    # TODO: Implement the function.                                           #
    ###########################################################################
    pass
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return n_nodes






