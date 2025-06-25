# tunner.py
import optuna
from optuna.samplers import TPESampler
from LightGBMModel import LightGBMModel
from scipy.stats import spearmanr
from sklearn.metrics import ndcg_score
import numpy as np
import pandas as pd
from custom_loss import exp_weight_rmse_loss,weighted_rmse_eval, compute_wpcc

class LightGBMTunner:
    def __init__(self, folds, X, y, search_space: dict, fixed_params: dict, quarter: str, n_trials: int = 50, n_startup_trials: int = 10, feval = None, eval_metric = None, mode: str = "ret_reg"):
        self.folds = folds  # list of (train_dates, val_dates)
        self.X = X
        self.y = y
        self.search_space = search_space
        self.fixed_params = fixed_params
        self.n_trials = n_trials
        self.n_startup_trials = n_startup_trials
        self.quarter = quarter
        self.eval_metric = eval_metric
        self.feval = feval
        self.mode = mode
    
    def _make_group(self, idx):
        """给定行索引列表或布尔数组，返回 group list"""
        sub = self.y.loc[idx].reset_index(level='Code', drop=True)
        return sub.groupby(level='date').size().tolist()

    def objective(self, trial):
        params = {
            **self.fixed_params,
            "learning_rate": trial.suggest_float("learning_rate", *self.search_space["learning_rate"]),
            "num_leaves": trial.suggest_int("num_leaves", *self.search_space["num_leaves"]),
            "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", *self.search_space["min_data_in_leaf"]),
            "lambda_l2": trial.suggest_float("lambda_l2", *self.search_space["lambda_l2"]),
            "subsample": trial.suggest_float("subsample", *self.search_space["subsample"]),
            "colsample_bytree": trial.suggest_float("colsample_bytree", *self.search_space["colsample_bytree"])
        }
        scores = []
        best_iters = []
        mode = self.mode
        eval_metric = self.eval_metric
        for train_dates, val_dates in self.folds:
            X_train = self.X.loc[self.X.index.get_level_values("date").isin(train_dates)]
            y_train = self.y.loc[X_train.index]
            X_val = self.X.loc[self.X.index.get_level_values("date").isin(val_dates)]
            y_val = self.y.loc[X_val.index]

            model = LightGBMModel(params)
            if mode == "rank_xendcg":
                mask_tr = self.X.index.get_level_values("date").isin(train_dates)
                mask_va = self.X.index.get_level_values("date").isin(val_dates)
                grp_tr = self._make_group(mask_tr)
                grp_va = self._make_group(mask_va)
                model = LightGBMModel(params)
                model.train(X_train = X_train, y_train = y_train, group_train=grp_tr, X_val=X_val, y_val=y_val, group_val=grp_va, best_iters=None)
                preds = model.predict(X_val)
                daily_ndcgs = []
                for date in y_val.index.get_level_values("date").unique():
                    # 用布尔掩码定位同一天的行
                    mask = y_val.index.get_level_values("date") == date
                    true = y_val[mask].values
                    pred = preds[mask]      # 用相同的布尔数组来切 numpy 数组
                    ndcg = ndcg_score([true], [pred], k=params.get("ndcg_eval_at", [1000])[0])
                    daily_ndcgs.append(ndcg)
                score = np.mean(daily_ndcgs)
                scores.append(score)
                best_iters.append(model.best_iteration)

            else:
                model.train(X_train = X_train, y_train = y_train, X_val = X_val, y_val=  y_val, feval=self.feval)
                preds = model.predict(X_val)
                df_val = pd.DataFrame({"pred": preds, "y": y_val}, index =X_val.index)
                if mode == "ret_reg" or "weightedret_reg":
                    if eval_metric == "wpcc":
                        dates = X_val.index.get_level_values("date")
                        daily_scores = []
                        for date in dates.unique():
                            mask = dates == date
                            daily_scores.append(compute_wpcc(df_val.loc[mask, "pred"], df_val.loc[mask, "y"]))
                        scores.append(np.mean(daily_scores))
                    else:
                        df_val["predNorm"] = df_val.groupby(level='date')['pred'].transform(lambda x: (x - x.mean()) / x.std(ddof=1))
                        ic = df_val.groupby(level = "date")[["predNorm", "y"]].corr().loc[:, "predNorm"].xs("y", level=1)
                        mean_ic = ic.mean()
                        scores.append(mean_ic)     
                elif mode == "rank_reg":
                    df_val["pred_rank"] = df_val.groupby(level = "date")["pred"].rank()
                    df_val['ret_rank'] = df_val.groupby(level='date')['y'].rank()
                    ic = df_val.groupby(level = "date")[["pred_rank", "ret_rank"]].corr().loc[:, "pred_rank"].xs("ret_rank", level=1)
                    mean_ic = ic.mean()
                    scores.append(mean_ic)  
            best_iters.append(model.best_iteration)

        trial.set_user_attr("best_iterations", best_iters)
        return np.mean(scores)

    def tune(self):
        sampler = TPESampler(n_startup_trials=self.n_startup_trials)
        study = optuna.create_study(direction="maximize", sampler = sampler, study_name=f"LightGBM_Tuning_{self.quarter}")
        study.optimize(self.objective, n_trials=self.n_trials)
        best_params = study.best_params
        best_iters = study.best_trial.user_attrs.get("best_iterations", [])
        best_params["best_iterations"] = best_iters
        return best_params
