import pandas as pd
import numpy as np
import xgboost as xgb

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

data = load_breast_cancer()
x_data = data.data
y_data = data.target

x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.2)
dtrain = xgb.DMatrix(x_train, y_train)
dtest = xgb.DMatrix(x_test[0:10], y_test[0:10])  # 只取10个样本测试，方便输出展示



# reg:logistic
params = {'objective': 'reg:logistic', 'booster': 'gbtree', 'max_depth': 3, 'silent':1}
watch_list = [(dtrain, 'train'), (dtest, 'eval')]
model = xgb.train(params=params, dtrain=dtrain, num_boost_round=5, evals=watch_list)

# 打印模型结构
model.dump_model('reg_logistic_model.txt')

# 预测：
# 模型转换后的输出值
ypred = model.predict(dtest)

# 模型原始的输出值（logistic输出前），叶子节点的权重值相加
ypred_margin = model.predict(dtest, output_margin=True)

# 原始输出值所在的叶子节点下标，将叶子节点的下标对应的值相加即为output_margin=True时的结果
ypred_leaf = model.predict(dtest, pred_leaf=True)

print('='*20, 'prob', '='*20)
print(ypred)
print('='*20, 'margin', '='*20)
print(ypred_margin)
print('='*20, 'leaf', '='*20)
print(ypred_leaf)