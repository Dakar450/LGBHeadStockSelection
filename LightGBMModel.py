# LightGBMModel.py
import lightgbm as lgb
import pandas as pd
from custom_loss import exp_weight_rmse_loss, weighted_rmse_eval, compute_wpcc

class LightGBMModel:
    def __init__(self, params):
        self.params = params
        self.model = None
        self.best_iteration = None

    def train(self, X_train, y_train, group_train = None, X_val=None, y_val=None, group_val = None, best_iters=None, feval=None):
        if self.params.get("objective") == "rank_xendcg" and group_train is not None:
            # 如果是排序任务，使用 group 参数
            train_data = lgb.Dataset(X_train, label=y_train, group=group_train)
            if X_val is not None and y_val is not None and group_val is not None:
                val_data = lgb.Dataset(X_val, label=y_val, group=group_val)
                self.model = lgb.train(
                    self.params,
                    train_data,
                    valid_sets=[train_data, val_data],
                    num_boost_round=1000,
                    callbacks=[lgb.early_stopping(stopping_rounds=10, verbose=True)]
                )
                self.best_iteration = self.model.best_iteration
            else:
                num_boost_round = best_iters if best_iters else 1000
                self.model = lgb.train(
                    self.params,
                    train_data,
                    num_boost_round=num_boost_round
                )
                self.best_iteration = num_boost_round
            train_data = lgb.Dataset(X_train, label=y_train)      
        else:
            # 如果是回归任务，不使用 group 参数
            train_data = lgb.Dataset(X_train, label=y_train)
            if X_val is not None and y_val is not None:
                # 如果提供验证集，则启用 early stopping
                val_data = lgb.Dataset(X_val, label=y_val)
                if feval is not None:
                # 如果提供了自定义损失函数和评估函数
                    self.model = lgb.train(
                        self.params,
                        train_data,
                        valid_sets=[train_data, val_data],
                        num_boost_round=1000,
                        feval=feval,
                        callbacks=[lgb.early_stopping(stopping_rounds=10, verbose=True)]
                    )
                else:
                    self.model = lgb.train(
                        self.params,
                        train_data,
                        valid_sets=[train_data, val_data],
                        num_boost_round=1000,
                        callbacks = [lgb.early_stopping(stopping_rounds=10, verbose=True), lgb.log_evaluation(50)]
                    )
                self.best_iteration = self.model.best_iteration
            else:
                # 如果没有验证集，使用外部传入的 early stop 平均轮数
                num_boost_round = best_iters if best_iters else 1000                
                self.model = lgb.train(
                    self.params,
                    train_data,
                    num_boost_round = num_boost_round
                )
                self.best_iteration = num_boost_round

    def predict(self, X):
        return self.model.predict(X, num_iteration=self.best_iteration)
