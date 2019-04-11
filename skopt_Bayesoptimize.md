## 使用skopt中Bayesian搜索寻找scikit-learn中算法的最优超参数
> [skopt](https://scikit-optimize.github.io/)是一个超参数优化库，包括随机搜索、贝叶斯搜索、决策森林和梯度提升树等，用于辅助寻找机器学习算法中的最优超参数。


### 上手实例
```python
from skopt import BayesSearchCV
from sklearn.datasets import load_digits
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split

X, y = load_digits(10, True)
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.75,random_state=0)

opt = BayesSearchCV(SVC(), 
		{'C': (1e-6, 1e+6, 'log-uniform'),
		'gamma': (1e-6, 1e+1, 'log-uniform'),
		'degree': (1,8), # 整数类型的空间 
		'kernel': ['linear','poly','rbf'], # Categorical类型的空间
		},
		n_iter=32
		)

opt.fit(X_train, y_train)
```



### 进阶实例
> 对多个模型及其对应的搜索空间
```python
from skopt import BayesSearchCV
from skopt.space import Real, Categorical, Integer  # 类：定义数据类型

from sklearn.datasets import load_digits
from sklearn.svm import LinearSVC, SVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

X, y = load_digits(10, True)
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.75, random_state=0)

pipe = Pipeline(
    [
        ('model', SVC())
    ])

# 隐式地指定参数类型
linsvc_search = {
    'model': [LinearSVC(max_iter=10000)],
    'model__C': (1e-6, 1e+6, 'log-uniform'),
}
# 显式地指定参数类型
svc_search = {
    'model': Categorical([SVC()]), 
    'model__C': Real(1e-6, 1e+6, prior='log-uniform'),  # 实值类型，下限和上限以及先验概率
    'model__gamma': Real(1e-6, 1e+1, prior='log-uniform'),
    'model__degree': Integer(1,8),
    'model__kernel': Categorical(['linear', 'poly', 'rbf'])  # 类型变量
}

opt = BayesSearchCV(
    pipe,
    [(svc_search, 20), (linsvc_search, 16)]    # 元组的含义：(搜索子空间，用于子空间中优化的迭代次数)
	)


opt.fit(X_train, y_train)

print("val. score: %s" % opt.best_score_)
print("test score: %s" % opt.score(X_test, y_test))


```
