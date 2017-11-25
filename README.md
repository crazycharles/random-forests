# random-forests

A basic RandomForest algorithm which was written in python. It uses the Gini index to split every node.
And I use the out-of-bag data to estimate the random forest's performance which can avoid much computation.

## Usage
```python
# load csv and transform it into a list.
dataset = load_csv('wine.data.csv')
# construct the random-forests.
rf = RandomForestClassifier(dataset, class_id=-1, max_depth=20, min_size=1, n_trees=500, n_features=False)
# output the oob test score.
print('oob_score:', rf.oob_score())
# output the test score on a test data set. I use the original data set here.
result = rf.predict(dataset)
print('predict score:', result['score'])
# output the feature importance sequence.
importance = rf.importance
print('feature importance:', importance)
```
## Parameters
     dataset:      list. You should transform your data to a list.  
     class_id:     int. The index of class you want to predict.  
     max_depth:    int. The maximum depth of the tree.  
     min_size:     int. The minimum number of samples required to split an internal node.  
     n_trees:      int. The number of tree you want to construct.  
     n_features:   int. The number of features to consider when looking for the best split. The default value is sqrt(length of varibles). 
 

## Reference
I implement this code based on Siraj Raval's code. But he used the cross-validation method to estimate the random forest's performance. It will be a long time if you train a large forest.  
https://github.com/llSourcell/random_forests
