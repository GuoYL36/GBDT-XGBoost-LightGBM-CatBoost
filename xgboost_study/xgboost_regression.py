import pandas as pd
import numpy as np
import xgboost as xgb

from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split

np.random.seed(1994)
data = load_boston()
x_data = data.data
y_data = data.target

x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.2)
dtrain = xgb.DMatrix(x_train, y_train)
dtest = xgb.DMatrix(x_test[0:10], y_test[0:10])

params = {'objective': 'reg:squarederror', 'booster': 'gbtree', 'max_depth': 3, 'silent':1}
watch_list = [(dtrain, 'train'), (dtest, 'eval')]

model = xgb.train(params=params, dtrain=dtrain, num_boost_round=5, evals=watch_list)

model.dump_model('reg_squarederror_model.txt')

ypred = model.predict(dtest)
ypred_margin = model.predict(dtest, output_margin=True) # 输出是叶子节点权重和加上偏置(0.5)
ypred_leaf = model.predict(dtest, pred_leaf=True)

print('='*20, 'prob', '='*20)
print(ypred)
print('='*20, 'margin', '='*20)
print(ypred_margin)
print('='*20, 'leaf', '='*20)
print(ypred_leaf)















