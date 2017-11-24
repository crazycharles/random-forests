# random-forests


A basic RandomForest algorithm which was written in python. It uses the Gini index to split every node.
And I use the out-of-bag data to estimate the random forest's performance which can avoid much computation.

## Usage
```python
dataset = load_csv('wine.data.csv')
rf = RandomForestClassifier(dataset, -1, max_depth=10, n_trees=500)
print('oob_score:',rf.oob_score())
result = rf.predict(dataset)
print('predict score:',result['score'])
```
## Reference
I implement this code based on Siraj Raval's code. But he used the cross-validation method to estimate the random forest's performance. It will be a long time if you train a large forest.

https://github.com/llSourcell/random_forests
