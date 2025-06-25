# main.py
import json
from multiprocessing import Process, Manager
import pandas as pd
from utils import DataLoader
import pickle
import os
from custom_loss import exp_weight_rmse_loss,weighted_rmse_eval, compute_wpcc


os.environ["OMP_NUM_THREADS"] = "20"
os.environ["OPENBLAS_NUM_THREADS"] = "20"
os.environ["MKL_NUM_THREADS"] = "20"
os.environ["NUMEXPR_NUM_THREADS"] = "20"


# 运行单个季度的流水线，包括调参、训练、回测，并收集结果
def run_one_quarter(test_quarter, data, gpu_id, config, factor_list):
    from LightGBMModel import LightGBMModel
    from TimeSplit import TimeSplit
    from tunner import LightGBMTunner
    from BuyStrategy import BuyStrategy


    # 时间切分：得到4折 folds、完整训练日期 all_train_dates、测试日期 test_dates
    data.sort_index(inplace = True)
    dates = data.index.get_level_values('date').unique()
    splitter = TimeSplit(test_quarter, dates)
    folds, all_train_dates, test_dates = splitter.get_split()
    
    # 加载损失函数等
    FUNC_MAP = {
    "weighted_rmse_eval":   weighted_rmse_eval,
    "wpcc":                 compute_wpcc,
    }
    mode = config.get("mode")
    eval_metric_name = config.get("eval_metric", None)
    feval_name = config.get("feval", None)
    eval_metric = FUNC_MAP.get(eval_metric_name, None)
    feval = FUNC_MAP.get(feval_name, None)
    
    # 加载当季数据
    X = data[factor_list]
    fixed_params = config["fixed_params"].copy()
    fixed_params["objective"] = {"rank_reg": "regression", "rank_xendcg": "rank_xendcg", "ret_reg": "regression", "weightedret_reg": exp_weight_rmse_loss}[mode]
    fixed_params["metric"] = {"rank_reg": "rmse", "ret_reg": "rmse", "rank_xendcg": "ndcg", "weightedret_reg": None}[mode]
    if mode == "ret_reg" or "weightedret_reg":
        y = data.groupby(level='date')['ret'].transform(lambda x: (x - x.mean()) / x.std(ddof=1))
    elif mode == "rank_reg":
        y = data.groupby(level = 'date')['ret'].rank(ascending=True, method = "average")
    elif mode == "rank_xendcg":
        fixed_params["ndcg_eval_at"] = [1000]
        quantiles = [0.0, 0.3, 0.5, 0.7, 0.8, 0.85, 0.9, 0.95, 1.0]
        labels = [0, 1, 2, 3, 4, 5, 6, 7]
        y = (data['ret'].groupby(level='date').transform(lambda x: pd.qcut(x, quantiles, labels=labels)).astype(int))
    y.name = "y_label"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)  # 设置当前进程使用的GPU
    # 调参或使用默认参数
    if config.get("tune", False):
        tuner = LightGBMTunner(folds, X, y, config["search_space"], fixed_params, quarter = test_quarter, n_startup_trials = config.get("n_startup_trials",20), n_trials=config.get("n_trials",35), eval_metric=eval_metric, feval=feval, mode = mode)
        best_params = tuner.tune()
    else:
        best_params = {**fixed_params, **config["default_params"]}
    # 提取并计算平均 early stop 轮数
    if best_params.get("best_iterations"):
        best_iters = best_params.pop("best_iterations", [])
    else:
        print("No best iterations found, using default boost round.")
        best_iters = [500]
    avg_iter = int(sum(best_iters) / len(best_iters)) if best_iters else config.get("default_boost_round",500)
    print(test_quarter, avg_iter, flush = True)
    
    # 5. 全量训练 (train+val)
    train_df = data.loc[data.index.get_level_values("date").isin(all_train_dates)]
    X_train, y_train = train_df[factor_list], y.loc[train_df.index]
    params = {**fixed_params, **best_params}
    model = LightGBMModel(params)
    if mode == "rank_xendcg":
        group_train = (y_train.reset_index(level='Code', drop=True).groupby(level='date').size().tolist())
        model.train(X_train = X_train, y_train = y_train, group_train = group_train, best_iters = avg_iter)
    else:
        model.train(X_train = X_train, y_train = y_train, best_iters = avg_iter, feval=feval)
    del X_train, y_train  # 释放内存

    # 6. 预测 & 排序
    test_df = data.loc[data.index.get_level_values("date").isin(test_dates)].copy()
    test_df["pred"] = model.predict(test_df[factor_list])
    test_result = test_df[["pred", "liquid", "ret"]]
    del test_df  # 释放内存

    # 7. 回测 & 记录
    stats = []
    for strategy_type in ["greedy", "equal", "liquid_equal"]:
        strat = BuyStrategy(test_result, top_pct=config["strategy"]["top_pct"], budget=config["strategy"]["budget"], strategy_type=strategy_type)
        perf = strat.run()
        stats.append(perf.assign(fold=test_quarter, strategy=strategy_type))

    # 8. 返回模型参数、绩效和选股列表
    result = {
        "test_quarter": test_quarter,
        "model_params": best_params,
        "best_iterations": best_iters,
        "stats": stats,
        "ModelResult": test_result
    }
    os.makedirs("TreeResults528wrmse", exist_ok=True)
    with open(f"TreeResults528wrmse/result_{test_quarter}.pkl", "wb") as f:
        pickle.dump(result, f)
    print(f"Completed {test_quarter}, Results collected.")


def main():
    # 读取配置
    with open("/home/user92/model_train/TreeModel/config.json") as f:
        config = json.load(f)
    folds = config["test_quarters"]
    with open(config["factor_list_path"]) as f:
        factor_list = json.load(f)
    
    loader = DataLoader(data_path=config["data_path"], fac_name=config["fac_name"], ret_name=config["ret_name"], liquid_name=config["liquid_name"])
    loader.load_full_data()
    
    gpu_ids = list(range(8))[::-1]  # 8个GPU设备，逆序使用


    procs = []
    for i, quarter in enumerate(folds):
        datai, factors = loader.get_dataset(factor_list[quarter], quarter)
        print(quarter)
        p = Process(
            target=run_one_quarter,
            args=(quarter, datai, gpu_ids[i], config, factors)
        )
        p.start()
        procs.append(p)
    del loader, datai, factors
    for p in procs:
        p.join()
        
    
    print("Completed all folds. Results saved to pkl files.")

if __name__ == "__main__":
    main()
