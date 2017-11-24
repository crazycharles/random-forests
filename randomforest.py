# Author:Chaojie An
# Github: https://github.com/crazycharles
# Date: 24, Nov 2017
# Note: This code for construct an original random forest classifier.
# Example:
#   dataset = load_csv('wine.data.csv')
#   rf = RandomForestClassifier(dataset, class_id=-1, max_depth=10, n_trees=500)
#   print(rf.oob_score)
# if you have a new data set to be classified, you can use:
#   s = rf.predict(new_dataset)
#   #accuracy
#   print(s['score'])
#   #output of prediction
#   print(s['output'])

from csv import reader
import random
import math


def error_detection(dataset,class_id,max_depth,min_size,n_trees,n_features):
    """This function to check the parameters's type right or wrong."""
    errors = list()
    if type(dataset) != list:
        errors.append("dataset should be a list.")
    if len(dataset) < 2:
        errors.append("The length of dataset should be greater than 2, or it is meaningless.")
    if type(class_id) != int:
        errors.append("class_id should be an integer.")
    if type(max_depth) != int or max_depth <= 0:
        errors.append("max_depth should be a positive integer.")
    if type(min_size) != int or min_size <= 0:
        errors.append("min_size should be a positive integer.")
    if type(n_trees) != int or n_trees <= 0:
        errors.append("n_trees should be a positive integer.")
    if type(n_features) != int or n_features <= 0:
        errors.append("n_features should be a positive integer.")
    if n_features > (len(dataset[0]) - 1):
        errors.append("variable length subtracts n_features should be greater than or equal to 1.")
    # Check the accuracy between class_id and dataset.
    # For example, the length of variables is 5, the class_id should be -5~4
    variables_length = len(dataset[0])
    if class_id < 0:
        if abs(class_id) > variables_length:
            errors.append("class_id Error, out of range!")
    else:
        if class_id > variables_length - 1:
            errors.append("class_id Error, out of range!")
    # Print the error information and stop here, if it exists.
    if len(errors) != 0:
        print(errors)
        return True
    else:
        return False


def load_csv(filename):
    """init the dataset as a list"""
    dataset = list()
    # open it as a readable file
    with open(filename, 'r') as file:
        # init the csv reader
        csv_reader = reader(file)
        # for every row in the dataset
        for row in csv_reader:
            if not row:
                continue
            # add that row as an element in our dataset list (2D Matrix of values)
            dataset.append(row)
    # return in-memory data matrix
    return dataset


def str_column_to_float(dataset, column):
    """Convert string column to float"""
    # iterate throw all the rows in our data matrix
    for row in dataset:
        # for the given column index, convert all values in that column to floats
        row[column] = float(row[column].strip())


def str_column_to_int(dataset, column):
    """Convert string column to integer"""
    # store a given column
    class_values = [row[column] for row in dataset]
    # create an unordered collection with no duplicates, only unique values
    unique = set(class_values)
    # init a lookup table
    lookup = dict()
    # for each element in the column
    for i, value in enumerate(unique):
        # add it to our lookup table
        lookup[value] = i
    # the lookup table stores pointers to the strings
    for row in dataset:
        row[column] = lookup[row[column]]
    # return the lookup table
    return lookup


def handle(dataset, class_id):
    """Transform original data set to a suitable set for next step. Such as transform the string to float or integer."""
    variables_length = len(dataset[0])
    variables_id = [i for i in range(variables_length)]
    variables_id.pop(class_id)
    for i in variables_id:
        str_column_to_float(dataset, i)
    str_column_to_int(dataset, class_id)
    return dataset


def bootstrap(dataset):
    """Sample the data set N times with reset, and N is the length of data set"""
    train = list()
    oob = list()
    n_sample = len(dataset)
    dataset_index = [i for i in range(n_sample)]
    oob_index = list()
    bootstrap_times = 0
    # To avoid the finite loop so plus the bootstrap_times condition.
    while len(oob) == 0 and bootstrap_times < 10:
        bootstrap_times = bootstrap_times + 1
        train = list()
        train_index = list()
        for i in range(n_sample):
            index = random.randrange(n_sample)
            train_index.append(index)
            train.append(dataset[index])
        ti = set(train_index)
        if len(ti) < n_sample:
            for i in ti:
                dataset_index.remove(i)
            for i in dataset_index:
                oob_index.append(i)
                oob.append(dataset[i])
    # The next line code almost never happen, but it guarantees the algorithm's safety.
    if bootstrap_times == 10:
        train = list()
        oob = list()
        if n_sample == 2:
            train.append(dataset[0])
            oob.append(dataset[1])
        else:
            j = int(0.368*n_sample)
            for i in range(j):
                oob.append(i)
            for i in range(j, n_sample):
                train.append(i)
    return {'train': train, 'oob': oob}


def test_split(index, value, dataset):
    """Split a dataset based on an attribute and an attribute value"""
    # init 2 empty lists for storing split datasubsets
    left, right = list(), list()
    # for every row
    for row in dataset:
        # if the value at that row is less than the given value
        if row[index] < value:
            # add it to list 1
            left.append(row)
        else:
            # else add it list 2
            right.append(row)
    # return both lists
    return left, right


def gini_index(groups, class_values, class_id):
    gini = 0.0
    # for each class
    for class_value in class_values:
        # a random subset of that class
        for group in groups:
            size = len(group)
            if size == 0:
                continue
            # average of all class values
            proportion = [row[class_id] for row in group].count(class_value) / float(size)
            #  sum all (p * 1-p) values, this is gini index
            gini += (proportion * (1.0 - proportion))
    return gini


def get_split(dataset, class_id, n_features):
    # Given a dataset, we must check every value on each attribute as a candidate split,
    # evaluate the cost of the split and find the best possible split we could make.
    class_values = list(set(row[class_id] for row in dataset))
    b_index, b_value, b_score, b_groups = 999, 999, 999, None
    features = list()
    # features_list should be a list without the class_id.
    features_list = [i for i in range(len(dataset[0]))]
    features_list.pop(class_id)
    while len(features) < n_features:
        index = random.choice(features_list)
        if index not in features:
            features.append(index)
    for index in features:
        index_values = list(set(row[index] for row in dataset))
        for value in index_values:
            # When selecting the best split and using it as a new node for the tree
            # we will store the index of the chosen attribute, the value of that attribute
            # by which to split and the two groups of data split by the chosen split point.
            # Each group of data is its own small dataset of just those rows assigned to the
            # left or right group by the splitting process. You can imagine how we might split
            # each group again, recursively as we build out our decision tree.
            groups = test_split(index, value, dataset)
            gini = gini_index(groups, class_values, class_id)
            if gini < b_score:
                b_index, b_value, b_score, b_groups = index, value, gini, groups
                # Once the best split is found, we can use it as a node in our decision tree.
                # We will use a dictionary to represent a node in the decision tree as
                # we can store data by name.
    return {'index': b_index, 'value': b_value, 'groups': b_groups}


def to_terminal(group, class_id):
    """Create a terminal node value"""
    # select a class value for a group of rows.
    outcomes = [row[class_id] for row in group]
    # returns the most common output value in a list of rows.
    return max(set(outcomes), key=outcomes.count)


# Create child splits for a node or make terminal
# Building a decision tree involves calling the above developed get_split() function over
# and over again on the groups created for each node.
# New nodes added to an existing node are called child nodes.
# A node may have zero children (a terminal node), one child (one side makes a prediction directly)
# or two child nodes. We will refer to the child nodes as left and right in the dictionary representation
# of a given node.
# Once a node is created, we can create child nodes recursively on each group of data from
# the split by calling the same function again.
def split(node, max_depth, min_size, n_features, depth, class_id):
    # Firstly, the two groups of data split by the node are extracted for use and
    # deleted from the node. As we work on these groups the node no longer requires access to these data.
    left, right = node['groups']
    del (node['groups'])
    # Next, we check if either left or right group of rows is empty and if so we create
    # a terminal node using what records we do have.
    # check for a no split
    if not left or not right:
        node['left'] = node['right'] = to_terminal(left + right, class_id)
        return
    # We then check if we have reached our maximum depth and if so we create a terminal node.
    # check for max depth
    if depth >= max_depth:
        node['left'], node['right'] = to_terminal(left, class_id), to_terminal(right, class_id)
        return
    # We then process the left child, creating a terminal node if the group of rows is too small,
    # otherwise creating and adding the left node in a depth first fashion until the bottom of
    # the tree is reached on this branch.
    # process left child
    if len(left) <= min_size:
        node['left'] = to_terminal(left, class_id)
    else:
        node['left'] = get_split(left, class_id, n_features)
        split(node['left'], max_depth, min_size, n_features, depth + 1, class_id)
    # process right child
    # The right side is then processed in the same manner,
    # as we rise back up the constructed tree to the root.
    if len(right) <= min_size:
        node['right'] = to_terminal(right, class_id)
    else:
        node['right'] = get_split(right, class_id, n_features)
        split(node['right'], max_depth, min_size, n_features, depth + 1, class_id)


def build_tree(train, class_id, max_depth, min_size, n_features):
    """Build a decision tree"""
    # Building the tree involves creating the root node and
    root = get_split(train, class_id, n_features)
    # calling the split() function that then calls itself recursively to build out the whole tree.
    split(root, max_depth, min_size, n_features, 1, class_id)
    return root


def choose(n_features):
    """This function  to choose the parameters, such as n_features, entropy function..."""
    # the default n_features equal to sqrt(variable_length).
    if not n_features:
        n_features = int(math.sqrt(len(dataset[0])))
    return n_features


def predict_oob(node, row):
    """Make a prediction with a decision tree"""
    # Making predictions with a decision tree involves navigating the
    # tree with the specifically provided row of data.
    # Again, we can implement this using a recursive function, where the same prediction routine is
    # called again with the left or the right child nodes, depending on how the split affects the provided data.
    # We must check if a child node is either a terminal value to be returned as the prediction
    # , or if it is a dictionary node containing another level of the tree to be considered.
    if row[node['index']] < node['value']:
        if isinstance(node['left'], dict):
            return predict_oob(node['left'], row)
        else:
            return node['left']
    else:
        if isinstance(node['right'], dict):
            return predict_oob(node['right'], row)
        else:
            return node['right']


def accuracy_metric(actual, predicted):
    """Calculate accuracy percentage"""
    # how many correct predictions?
    correct = 0
    # for each actual label
    for i in range(len(actual)):
        # if actual matches predicted label
        if actual[i] == predicted[i]:
            # add 1 to the correct iterator
            correct += 1
    # return percentage of predictions that were correct
    return correct / float(len(actual)) * 100.0


def oob_test(tree, oob_sample, class_id):
    actual = [row[class_id] for row in oob_sample]
    predictions = [predict_oob(tree, row) for row in oob_sample]
    oob_score = accuracy_metric(actual, predictions)
    return oob_score


class Vector:
    def __init__(self, trees, trees_accuracy, class_id):
        self.trees = trees
        self.trees_accuracy = trees_accuracy
        self.class_id = class_id

    def oob_score(self):
        scores = self.trees_accuracy
        scores = sum(scores)*1.0/len(scores)
        return scores

    def predict(self, test_set):
        class_id = self.class_id
        trees = self.trees
        actual = [row[class_id] for row in test_set]
        predictions = list()
        for row in test_set:
            outcomes = [predict_oob(tree, row) for tree in trees]
            predictions.append(max(set(outcomes), key=outcomes.count))
        score = accuracy_metric(actual, predictions)
        return {'score': score, 'output': predictions}


def RandomForestClassifier(dataset=list(), class_id=-1, max_depth=100, min_size=1, n_trees=20, n_features=False):
    n_features = choose(n_features)
    # the detection step.
    if error_detection(dataset, class_id, max_depth, min_size, n_trees, n_features):
        return
    # handle the data set.
    dataset = handle(dataset, class_id)
    trees = list()
    trees_accuracy = list()
    # construct the decision tree successively.
    for i in range(n_trees):
        sample = bootstrap(dataset)
        train_sample = sample['train']
        oob_sample = sample['oob']
        tree = build_tree(train_sample, class_id, max_depth, min_size, n_features)
        accuracy = oob_test(tree, oob_sample, class_id)
        trees.append(tree)
        trees_accuracy.append(accuracy)
    return Vector(trees=trees, trees_accuracy=trees_accuracy, class_id= class_id)


dataset = load_csv('wine.data.csv')
rf = RandomForestClassifier(dataset, -1, max_depth=10, n_trees=500)
print('oob_score:',rf.oob_score())
result = rf.predict(dataset)
print('predict score:',result['score'])
