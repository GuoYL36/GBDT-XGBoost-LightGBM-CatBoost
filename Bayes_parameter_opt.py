# tuning hyperparameters
'''
    使用贝叶斯优化对lightgbm进行调参
'''

from bayes_opt import BayesianOptimization
from skopt import BayesSearchCV

# building models
import lightgbm as lgb


def bayes_parameter_opt_lgb(X, y, init_round=15, opt_round=25, n_folds=3, random_seed=6, n_estimators=10000,
                            output_process=False):
    # prepare data
    train_data = lgb.Dataset(data=X, label=y, free_raw_data=False)

    # parameters
    def lgb_eval(learning_rate, num_leaves, feature_fraction, bagging_fraction, max_depth, max_bin, min_data_in_leaf,
                 min_sum_hessian_in_leaf, subsample):
        params = {'application': 'binary', 'metric': 'auc'}
        params['learning_rate'] = max(min(learning_rate, 1), 0)
        params['num_leaves'] = int(round(num_leaves))
        params['feature_fraction'] = max(min(feature_fraction, 1), 0)
        params['bagging_fraction'] = max(min(bagging_fraction, 1), 0)
        params['max_depth'] = int(round(max_depth))
        params['max_bin'] = int(round(max_bin))
        params['min_data_in_leaf'] = int(round(min_data_in_leaf))
        params['min_sum_hessian_in_leaf'] = min_sum_hessian_in_leaf
        params['subsample'] = max(min(subsample, 1), 0)

        cv_result = lgb.cv(params, train_data, nfold=n_folds, seed=random_seed, stratified=True, verbose_eval=200,
                           metrics=['auc'])

        return max(cv_result['auc-mean'])

    lgb0 = BayesianOptimization(lgb_eval,
                                {'learning_rate': (0.01, 1.0), 'num_leaves': (24, 80), 'feature_fraction': (0.1, 0.9),
                                 'bagging_fraction': (0.8, 1.0), 'max_depth': (5, 30), 'max_bin': (20, 90),
                                 'min_data_in_leaf': (20, 80), 'min_sum_hessian_in_leaf': (0, 100),
                                 'subsample': (0.01, 1.0)}, random_state=200)

    # n_iter：How many steps of bayesian optimization you want to perform. The more steps the more likely to find a good maximum you are.
    # init_points：How many steps of random exploration you want to perform. Random exploration can help by diversifying the exploration space.

    lgb0.maximize(n_iter=opt_round, init_points=init_round)

    model_auc = []
    for model in range(len(lgb0.res)):
        model_auc.append(lgb0.res[model]['target'])

    # return best parameters
    return lgb0.res[pd.Series(model_auc).idxmax()]['target'], lgb0.res[pd.Series(model_auc).idxmax()]['params']


opt_params = bayes_parameter_opt_lgb(X, y, init_round=5, opt_round=10, n_folds=3, random_seed=6, n_estimators=10000)

# Here is optimal parameter for lightgbm

opt_params[1]["num_leaves"] = int(round(opt_params[1]["num_leaves"]))
opt_params[1]['max_depth'] = int(round(opt_params[1]['max_depth']))
opt_params[1]['min_data_in_leaf'] = int(round(opt_params[1]['min_data_in_leaf']))
opt_params[1]['max_bin'] = int(round(opt_params[1]['max_bin']))
opt_params[1]['objective'] = 'binary'
opt_params[1]['metric'] = 'auc'
opt_params[1]['is_unbalance'] = True
opt_params[1]['boost_from_average'] = False
opt_params = opt_params[1]
# opt_params






