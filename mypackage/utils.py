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

def weight_calc(data,product):

    # calculate the denominator of RMSSE, and calculate the weight base on sales amount
    
    sales_train_val = pd.read_csv('inputs/sales_train_validation.csv')
    
    d_name = ['d_' + str(i+1) for i in range(1913)]
    
    sales_train_val = weight_mat_csr * sales_train_val[d_name].values
    
    # calculate the start position(first non-zero target observed date) for each item / 商品の最初の売上日
    # 1-1914のdayの数列のうち, 売上が存在しない日を一旦0にし、0を9999に置換。そのうえでminimum numberを計算
    df_tmp = ((sales_train_val>0) * np.tile(np.arange(1,1914),(weight_mat_csr.shape[0],1)))
    
    start_no = np.min(np.where(df_tmp==0,9999,df_tmp),axis=1)-1
    
    
    # denominator of RMSSE / RMSSEの分母
    weight1 = np.sum((np.diff(sales_train_val,axis=1)**2),axis=1)/(1913-start_no)
    
    # calculate the sales amount for each item/level
    df_tmp = data[(data['date'] > '2016-03-27') & (data['date'] <= '2016-04-24')]
    df_tmp['amount'] = df_tmp['target'] * df_tmp['sell_price']
    df_tmp =df_tmp.groupby(['id'])['amount'].apply(np.sum).values
    
    weight2 = weight_mat_csr * df_tmp 

    weight2 = weight2/np.sum(weight2)
    
    del sales_train_val
    gc.collect()
    
    return weight1, weight2

def submit():
    os.chdir("submissions")
    os.system("kaggle competitions submit -c m5-forecasting-accuracy -f submission.csv -m lgb")
    # kaggle competitions submit -c m5-forecasting-accuracy -f submission.csv -m lgb
    print(" 제출 완료")
    os.chdir("../")
 
def wrmsse(preds, data):
    
    # actual obserbed values / 正解ラベル
    y_true = data.get_label()
    
    # number of columns
    num_col = len(y_true)//NUM_ITEMS
    
    # reshape data to original array((NUM_ITEMS*num_col,1)->(NUM_ITEMS, num_col) ) / 推論の結果が 1 次元の配列になっているので直す
    reshaped_preds = preds.reshape(num_col, NUM_ITEMS).T
    reshaped_true = y_true.reshape(num_col, NUM_ITEMS).T
    
    x_name = ['pred_' + str(i) for i in range(num_col)]
    x_name2 = ["act_" + str(i) for i in range(num_col)]
          
    train = np.array(weight_mat_csr*np.c_[reshaped_preds, reshaped_true])
    
    score = np.sum(
                np.sqrt(
                    np.mean(
                        np.square(
                            train[:,:num_col] - train[:,num_col:])
                        ,axis=1) / weight1) * weight2)
    
    return 'wrmsse', score, False
 
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