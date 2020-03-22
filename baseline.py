import pandas as pd
import numpy as np 
from lightgbm import LGBMRegressor
from sklearn import metrics
from sklearn.model_selection import  KFold
import pickle
import time
import datetime
import os

# pd.set_option('display.max_columns', None)
# pd.set_option('display.max_rows', None)
# pd.set_option('display.max_colwidth', None)

def reduce_mem_usage(df, verbose=True):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024**2    
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)    
    end_mem = df.memory_usage().sum() / 1024**2
    if verbose: print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))
    return df

def write_record(features, params):
    record = open("record_model_and_features.txt", 'a')
    record.write("\n")
    record.write(str(datetime.datetime.now())+"\n")

    check = 0
    for _ in features:
        check += 1
        if check % 5 == 0:
            record.write("\n")
        record.write(_+"  ")
    record.write("\n")
    for i  in params.items():
        record.write(str(i) + "\n")

    record.write('--------------------------------\n')
    record.close()


print('data loaded')
with open('inputs/all_df.pickle', 'rb') as f:
    all_df = pickle.load(f)


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
        n_estimators =5000,
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
write_record(features, params)