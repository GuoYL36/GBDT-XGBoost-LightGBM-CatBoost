import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split

import numpy as np


# 参考https://www.jianshu.com/p/2920c97e9e16或https://blog.csdn.net/sujinhehehe/article/details/84201415

sample_num = 10
feature_num = 2

np.random.seed(0)
data = np.random.randn(sample_num, feature_num)
np.random.seed(0)
label = np.random.randint(0, 2, sample_num)

print("="*6)
print(data)
print("="*6)

print("="*6)
print(label)
print("="*6)

train_data = xgb.DMatrix(data, label=label)
params = {'max_depth': 3}
bst = xgb.train(params, train_data, num_boost_round=1)



for importance_type in ('weight', 'gain', 'cover', 'total_gain', 'total_cover'):
    print('%s: ' % importance_type, bst.get_score(importance_type=importance_type))


# 打印模型结构
# bst.dump_model('xgboost_model_featureImportance.txt')
trees = bst.get_dump(with_stats=True)
for tree in trees:
    print(tree)

# 从上述打印的结果可以看出:
#     weight：特征作为分裂节点出现的次数
#     gain：特征作为分裂节点的总增益除以出现次数
#     total_gain：特征作为分裂节点的总增益
#     cover：特征作为分裂节点时，覆盖的样本总数除以出现次数
#     total_cover：特征作为分裂节点时，覆盖的样本总数



























