import pandas as pd
import numpy as np
import xgboost as xgb

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

np.random.seed(1994)

kRows = 100
kCols = 10
kClasses = 4                    # number of classes
kRounds = 10                    # number of boosting rounds.

# Generate some random data for demo.
x_data = np.random.randn(kRows, kCols)
y_data = np.random.randint(0, 4, size=kRows)

x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.05, random_state=1)
print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)

dtrain = xgb.DMatrix(x_train, y_train)
dtest = xgb.DMatrix(x_test, y_test)


params = {'objective': 'multi:softprob', 'num_class': 4, 'booster': 'gbtree', 'max_depth': 3, 'silent':1}
watch_list = [(dtrain, 'train'), (dtest, 'eval')]

# total number of built trees is num_parallel_tree * num_classes * num_boost_round
model1 = xgb.train(params=params, dtrain=dtrain, num_boost_round=3, evals=watch_list)

model1.dump_model('multi_softmax_model.txt')

ypred = model1.predict(dtest)

ypred_margin = model1.predict(dtest, output_margin=True)  # shape=[20,4]，输出是叶子节点权重和加上偏置(0.5)，也是每个特征贡献值加上偏置
ypred_leaf = model1.predict(dtest, pred_leaf=True)  # shape=[20,3*4]，num_boost_round*num_class，相当于在迭代过程中，有4路在并行训练，对应每个类别，每个类别对应有3棵树

# 输出为[nSamples, nClass, nFeature+bias_term]，最后一列为bias，剩余列表示每个特征的贡献值(shap值)
ypred_contribs = model1.predict(dtest, pred_contribs=True)  # shape=[20, 4, 11]，最后一维求和后的结果等于参数output_margin的结果

print('='*20, 'prob', '='*20)
print(ypred)
print('='*20, 'margin', '='*20)
print(ypred_margin)
print('='*20, 'leaf', '='*20)
print(ypred_leaf)
print('='*20, 'pred_contribs', "="*20)
print(ypred_contribs)
