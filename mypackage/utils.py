import os
import kaggle
import numpy as np
import datetime
import pandas as pd


def reduce_memory(df, verbose=True):
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


def submit():
    os.chdir("submissions")
    os.system("kaggle competitions submit -c m5-forecasting-accuracy -f submission.csv -m lgb")
    # kaggle competitions submit -c m5-forecasting-accuracy -f submission.csv -m lgb
    print(" 제출 완료")
    os.chdir("../")
    
def write_params_features(features, params, eval_results):
    record = open("record_model_and_features.txt", 'a')
    record.write("\n")
    record.write(str(datetime.datetime.now())+"\n")

    check = 0
    for _ in features:
        check += 1
        if check % 10 == 0:
            record.write("\n")
        record.write(_+"  ")
    record.write("\n")
    for i  in params.items():
        record.write(str(i) + "\n")

    check = 0
    for row in eval_results:
        check += 1
        if check % 20 == 0:
            record.write("\n")
        record.write(str(row) + '  ')

    record.write('\n--------------------------------\n')
    record.close()

def save_feature_importance(feature):
    fi = pd.read_csv('feature_importances.csv')
    tmp = pd.DataFrame({'feature': ['---'], 'fold_1': ['---'], 'fold_2': ['---'], 'fold_3': ['---'], 'fold_4': ['---'], 'fold_5': ['---']})
    fi = pd.concat([fi, tmp])
    fi = pd.concat([fi, feature])
    fi.to_csv('feature_importances.csv', index=False)
    pass