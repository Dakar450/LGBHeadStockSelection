import sys
import os
import warnings
from argparse import ArgumentParser
from tqdm import tqdm
from lightgbm import LGBMRegressor, LGBMRanker
from xgboost import XGBRegressor
from lightgbm import log_evaluation, early_stopping
import joblib
import gc
import pandas as pd
import numpy as np
import shutil
import copy
import pickle
from datetime import datetime
from dateutil.relativedelta import relativedelta
from pytorch_lightning import Trainer, seed_everything
warnings.filterwarnings("ignore")
from joblib import Parallel, delayed 
import lightgbm as lgb
import gc
import optuna


def get_basic_name(model_prefix):
    name = rf'{model_prefix}--{fac_name}--{label_name}'
    return name

# 模型文件存储目录
root_path =rf'/home/user92/model_train/TreeModel'
# 数据存储目录
data_path = rf'/project/model_share/share_1'
# 因子
fac_path = rf'{data_path}/factor_data1'
fac_name = rf'fac20250508_528'
# 标签
label_path = rf'{data_path}/label_data'
label_name = rf'label1'
# 流动性数据
liquid_path = rf'{data_path}/label_data'
liquid_name = rf'can_trade_amt1'

def zscore_col(data):
    def func(data_X):
        data_X = np.clip(data_X, data_X.quantile(0.005), data_X.quantile(0.995))
        data_X = (data_X - data_X.mean()) / data_X.std()
        data_X = data_X.fillna(0)
        return data_X
    return data.groupby('date').apply(func)


def process_cs(data_X):
    data_X = np.sign(data_X)*np.log(1+np.abs(data_X))
    data_X = np.clip(data_X, data_X.quantile(0.005), data_X.quantile(0.995), axis=1)
    #data_X = (data_X - data_X.mean()) / data_X.std()
    #data_X = data_X.fillna(0)
    data_X = data_X.fillna(data_X.mean()) 
    return data_X


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--tmpfs_fast', default=True)
     
    parser.add_argument('--max_epochs', type=int, default=8)
    parser.add_argument('--workers', type=int, default=10)
    parser.add_argument('--accelerator', default='gpu')
    parser.add_argument('--gpu_id', default=0)
    parser.add_argument('--devices', type=int, default=4)
    parser.add_argument('--strategy',default='ddp')
    parser.add_argument('--find_unused_parameters',default=False)
    parser.add_argument('--threads', type=int, default=10)
    parser.add_argument('--n_jobs', type=int, default=10)
    parser.add_argument('--persistent_workers', default=False)
    parser.add_argument('--seed', type=int, default=42) 
    
    parser.add_argument('--objective', default='reg:squarederror', help='regression, rank')
    parser.add_argument('--metric', default='rmse')

    parser.add_argument('--booster', default='gbtree')
    parser.add_argument('--tree_method', default='gpu_hist')
    parser.add_argument('--max_depth', type=int, default=7) 
    parser.add_argument('--n_estimators', type=int, default=6053)
    parser.add_argument('--reg_lambda', type=float, default=5e-3)
    parser.add_argument('--lambda_l2', type=float, default=5e-3) 

    parser.add_argument('--learning_rate', type=float, default=0.014)
    parser.add_argument('--subsample', type=float, default=0.8) 
    parser.add_argument('--colsample_bytree', type=float, default=0.9)
    parser.add_argument('--min_child_weight', type=float, default=297) 

    parser.add_argument('--verbose', type=int, default=-1)
    parser.add_argument('--batch_size', type=int, default=1)
    args, unknown = parser.parse_known_args(args=[])
    return args


def train(args, name, market, season, state='train', k=1):
    # 保存最终模型的路径
    save_path = rf"{root_path}/model_test/{get_basic_name(name)}"
    model_name = f"{save_path}/{name[:len(market) + 19]}"
    test_save_path = f"{save_path}/{name[:len(market)] + name[len(market) + 6:len(market) + 19]}"
    os.makedirs(test_save_path, exist_ok=True)

    # 读取因子
    sel_fac_name_save = [x for x in os.listdir(rf'{fac_path}/{fac_name}') if season in x]
    assert len(sel_fac_name_save) == 1
    sel_fac_name_save = sel_fac_name_save[0]
    all_data = pd.read_feather(rf'{fac_path}/{fac_name}/{sel_fac_name_save}')
    ret_data = pd.read_feather(rf"{label_path}/{label_name}.fea").set_index("index")
    ret_data = ret_data.stack()
    ret_data.index.names = ['date', 'Code']
    ret_data = pd.DataFrame(ret_data, columns=['Label'])

    liquid_data =  pd.read_feather(rf"{liquid_path}/{liquid_name}.fea").set_index("index")
    date_list = list(all_data["date"].unique())
    date_list = [x for x in date_list if x in ret_data.index.get_level_values(0).unique() and x in liquid_data.index]
    date_list.sort()
     #### 修改
    all_data = all_data.set_index(["date", 'Code']).sort_index()
    all_data.dropna(thresh=int(0.1 * len(all_data.columns)), inplace=True)

    def get_train_date_split(season, period):
        test_start = season[:4] + str(int(season.split("q")[1]) * 3 - 2).zfill(2)
        start_date = datetime.strptime(test_start, "%Y%m")
        valid_date_split = []
        for i in [-3, 0, 6, 12, 18, 24]:
            valid_date_split.append((start_date - relativedelta(months=i)).strftime("%Y%m"))
        valid_date_split.reverse()
        train_start = valid_date_split[0]
        valid_start = valid_date_split[period - 1]
        valid_end = valid_date_split[period]
        test_end = valid_date_split[-1]
        return train_start, valid_start, valid_end, test_start, test_end

    train_start, valid_start, valid_end, test_start, test_end = get_train_date_split(season, k)

    if train_start < "202101":
        train_start = "202101"

    # 获取训练集，验证集，测试集日期
    # 隔开10天以防泄露未来信息
    valid_date_list = [x for x in date_list if valid_start <= x < valid_end][:-10]
    train_date_list = [x for x in date_list if train_start <= x < valid_start][:-10] + [x for x in date_list if
                                                                                        valid_end <= x < test_start][
                                                                                       10:-10]

    test_date_list = [x for x in date_list if test_start <= x < test_end]
    # 极端行情不参与训练
    not_train_date = [x for x in date_list if (x >= "202402") & (x <= "20240223")]
    train_date_list = [x for x in train_date_list if x not in not_train_date]

    all_data = all_data.loc[train_date_list+valid_date_list+test_date_list, :]
    all_data = all_data.groupby('date').apply(process_cs) 

    if len(test_date_list) == 0:
        test_date_list = ['out_sample']
    elif market == 'ALL':
        all_data = all_data.loc[:, all_data.replace(0, np.nan).dropna(how="all", axis=1).columns]
    else:
        raise NotImplementedError


    feature_map = list(all_data.columns[1:])
    factor_list = feature_map[:]
    with open(rf'{save_path}/{market}{name[len(market):len(market) + 6]}-feature_map.fea', 'w') as file:
        for idx, factor_name in enumerate(feature_map):
            file.write(rf'{factor_name}={idx}')
            file.write('\n')

    factor_num = all_data.shape[1] - 1


    

    if state == 'train':
        seed_everything(args.seed)
        all_data['Label'] = ret_data.reindex(all_data.index).values

        all_data['Label'] = all_data['Label'].groupby('date').apply(lambda x:((x-x.mean())/x.std()))#.apply(lambda x:(x.rank(pct=True)-0.5)*3.46)# .clip(-5,5)
        all_data.dropna(subset=['Label'], inplace=True)
        train_x = all_data.loc[train_date_list, factor_list].values
        train_y = all_data.loc[train_date_list,'Label'].values.squeeze()
        
        valid_x = all_data.loc[valid_date_list, factor_list].values
        valid_group = all_data.loc[valid_date_list, factor_list].index.get_level_values(0)
        valid_group = np.array([int(i) for i in valid_group])
        valid_y = all_data.loc[valid_date_list,'Label'].values.squeeze()
        test_x = all_data.loc[test_date_list, factor_list]

        del all_data
        gc.collect()

        def eval_score_ic(y_true, y_pred):
        # 求分组IC均值
            if type(y_pred) == lgb.Dataset:
                y_pred_array = y_pred.get_label()
            else:
                y_pred_array = y_pred
            
            y_df = pd.DataFrame({'y_true':y_true,'y_pred':y_pred_array,'group':valid_group})
            y_df = y_df.groupby('group').apply(lambda x: x.corr()['y_true']['y_pred'])
            score = y_df.mean()

            return -score

        
        def eval_score_ric(y_true, y_pred):
        # 求分组IC均值
            if type(y_pred) == lgb.Dataset:
                y_pred_array = y_pred.get_label()
            else:
                y_pred_array = y_pred
            
            y_df = pd.DataFrame({'y_true':y_true,'y_pred':y_pred_array,'group':valid_group})
            y_df = y_df.groupby('group').apply(lambda x: x.corr(method='spearman')['y_true']['y_pred'])
            score = y_df.mean()

            return -score

        def eval_score(y_true, y_pred):
          
            if type(y_pred) == lgb.Dataset:
                y_pred_array = y_pred.get_label()
            else:
                y_pred_array = y_pred
            
            y_df = pd.DataFrame({'y_true':y_true,'y_pred':y_pred_array,'group':valid_group})
            y_df = y_df.groupby('group').apply(lambda x: x.nlargest(int(len(x)*0.15),'y_pred')['y_true'].mean())
            score = y_df.mean()
            
            return -score


        train_params = {
            'objective': args.objective,
            'eval_metric': args.metric,
            'booster': args.booster,
            'max_depth': args.max_depth,
            'learning_rate': args.learning_rate,
            'reg_lambda': args.reg_lambda,
            'subsample': args.subsample,
            'colsample_bytree': args.colsample_bytree,
            'n_estimators': args.n_estimators,
            'tree_method':args.tree_method,
            'min_child_weight':args.min_child_weight, 
            'single_precision_histogram':False, 
            # 训练设置
            'random_state': args.seed,
            'n_jobs': args.n_jobs,
            'early_stopping_rounds': 50,  # 直接集成到参数中
            'verbosity': 1 if args.verbose else 0,
            'predictor': 'gpu_predictor',
            'gpu_id':args.gpu_id,
            'eval_metric': eval_score_ic
        }

       
        model = XGBRegressor(**train_params)

        model.fit(
            train_x, train_y,
            eval_set=[(valid_x, valid_y)],
            verbose=5,  # 每5轮输出日志
            #num_boost_round=1000
        )
        y_pred = model.predict(test_x, iteration_range=(0, model.best_iteration))
        return pd.DataFrame(y_pred, index=test_x.index, columns=['score'])
    
    
args = parse_args()
args.seed = 2964
args.max_depth = 9
args.learning_rate = 0.005
k =  int(sys.argv[1])
args.gpu_id = k+3
print(k)

res = []
for season in ['2023q1','2023q2', '2023q3','2023q4','2024q1','2024q2']:
    ypred = train(args, 'XGB_test', market="ALL", season=season, k=k)
    res.append(ypred)

res = pd.concat(res)
res.to_pickle(f'/home/user92/model_train/TreeModel/XGB4_s{args.seed}_{k}.pkl')

