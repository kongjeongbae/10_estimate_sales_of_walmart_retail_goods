import pandas as pd
import numpy as np 
from lightgbm import LGBMRegressor
from sklearn import metrics
from sklearn.model_selection import  KFold
import pickle
import time
import datetime
import os
from functions import *
# pd.set_option('display.max_columns', None)
# pd.set_option('display.max_rows', None)
# pd.set_option('display.max_colwidth', None)

print('data loading')
with open('inputs/all_df.pickle', 'rb') as f:
    all_df = pickle.load(f)
print('data loaded')

# model setting
print('model setting')

features = ['item_id', 'dept_id', 'cat_id', 'store_id', 'state_id', 'year', 'month', 'week', 'day', 'weekday', 'event_name_1', 'event_type_1',  
            'snap_CA', 'snap_TX', 'snap_WI', 'sell_price', 'lag_t28', 'lag_t29', 'lag_t30', 'rolling_mean_t7', 'rolling_std_t7', 'rolling_mean_t30', 'rolling_mean_t90', 
            'rolling_mean_t180', 'rolling_std_t30', 'price_change_t1', 'price_change_t365', 'rolling_price_std_t7', 'rolling_price_std_t30']

# 나중에 합칠 때 필요해서 test에 선언
test = all_df[58327370:]

train_set_X = all_df[:58327370]
train_set_y = train_set_X['target']

train_set_X = train_set_X[features]

# 테스트 셋
test_set = all_df[58327370:]
test_set = test_set[features]

del all_df

# model run
n_fold = 5
folds = KFold(n_splits=5, shuffle=True)
splits = folds.split(train_set_X, train_set_y)

y_preds = np.zeros(test.shape[0])
y_oof = np.zeros(train_set_X.shape[0])

feature_importances = pd.DataFrame()
feature_importances['feature'] = train_set_X.columns
mean_score = []

for fold_n, (train_index, valid_index) in enumerate(splits):
    print('Fold:',fold_n+1)
    
    X_train, X_valid = train_set_X.iloc[train_index], train_set_X.iloc[valid_index]
    y_train, y_valid = train_set_y.iloc[train_index], train_set_y.iloc[valid_index]
    
    lgb = LGBMRegressor(
        boosting_type = 'gbdt',
        num_leaves = 4000,
        colsample_bytree = 0.8,
        subsample = 0.8,
        n_estimators =600,
        learning_rate = 0.2,
        n_jobs = -1,
        device = 'gpu'
    )
    lgb.fit(X_train, y_train, eval_set=[(X_valid, y_valid)], early_stopping_rounds = 50, verbose = True)
    
    # 피쳐중요도 작성
    feature_importances[f'fold_{fold_n + 1}'] = lgb.feature_importances_
    
    # validation predict
    y_pred_valid = lgb.predict(X_valid, num_iteration=lgb.best_iteration_)

    y_oof[valid_index] = y_pred_valid
    
    val_score = np.sqrt(metrics.mean_squared_error(y_pred_valid, y_valid))
    
    print(f'val rmse score is {val_score}')
    
    mean_score.append(val_score)
    
    y_preds += lgb.predict(test_set, num_iteration=lgb.best_iteration_) / n_fold
    
    del X_train, X_valid, y_train, y_valid

print('mean rmse score over folds is',np.mean(mean_score))
test['target'] = y_preds

# predict
sub = pd.read_csv('inputs/sample_submission.csv')

predictions = test[['id', 'date', 'target']]
predictions = pd.pivot(predictions, index = 'id', columns = 'date', values = 'target').reset_index()
predictions.columns = ['id'] + ['F' + str(i + 1) for i in range(28)]

evaluation_rows = [row for row in sub['id'] if 'evaluation' in row] 
evaluation = sub[sub['id'].isin(evaluation_rows)]

validation = sub[['id']].merge(predictions, on = 'id')
final = pd.concat([validation, evaluation])
final.to_csv('submissions/submission.csv', index = False)

features = train_set_X.columns
params = lgb.get_params()
write_params_features(features, params)