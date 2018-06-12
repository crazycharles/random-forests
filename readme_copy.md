# random-forests

A basic RandomForest algorithm which was written in python. It uses the Gini index to split every node.
By means of the out-of-bag data to estimate the random forest's performance which can avoid the time-consuming cross-validation.

## Background
Random forests is an ensembling learning method. It consists of a set of decision trees. So it has significant performance on classification or regression tasks.
If you are not familiar with the random-forests algorithm, you can check the following information for more details:  
     **Site:**  
     1.Wiki: https://en.wikipedia.org/wiki/Random_forest  
     **Paper:**  
     2.Ho, Tin Kam (1995). [Random Decision Forests](http://ect.bell-labs.com/who/tkh/publications/papers/odt.pdf)(PDF). Proceedings of the 3rd International Conference on Document Analysis and Recognition, Montreal, QC, 14–16 August 1995. pp. 278–282.  
     3.Ho, Tin Kam (1998). [The Random Subspace Method for Constructing Decision Forests](http://ect.bell-labs.com/who/tkh/publications/papers/df.pdf)(PDF). IEEE Transactions on Pattern Analysis and Machine Intelligence. 20 (8): 832–844. doi:10.1109/34.709601.  
     4.Breiman, Leo (2001).[Random Forests](https://link.springer.com/article/10.1023%2FA%3A1010933404324)(PDF). Machine Learning. 45 (1): 5–32. doi:10.1023/A:1010933404324.  
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
