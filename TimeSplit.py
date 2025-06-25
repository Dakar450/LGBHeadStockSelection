from datetime import datetime
from dateutil.relativedelta import relativedelta
import pandas as pd

class TimeSplit:
    def __init__(self, season: str, date_list: list, min_train_date='2021-01-01'):
        self.season = season
        self.date_list = sorted([
            pd.to_datetime(d) if not isinstance(d, datetime) else d
            for d in date_list
        ])
        q = pd.Period(season, freq='Q')
        self.min_train_date = pd.to_datetime(min_train_date)
        self.test_start = q.start_time.normalize()
        self.test_end = q.end_time.normalize()
        start_date = self.test_start
        self.valid_date_split = [
            start_date - relativedelta(months=24),
            start_date - relativedelta(months=18),
            start_date - relativedelta(months=12),
            start_date - relativedelta(months=6),
            start_date
        ]
        self.not_train_date = [d for d in self.date_list if pd.to_datetime("20240201") <= d <= pd.to_datetime("20240223")]

    def get_split(self, gap_days=10):
        folds = []
        for i in range(4):
            train_start = self.valid_date_split[0]
            valid_start = self.valid_date_split[i]
            valid_end = self.valid_date_split[i + 1]

            if train_start < self.min_train_date:
                train_start = self.min_train_date

            train_dates = [x for x in self.date_list if train_start <= x < valid_start][:-gap_days] + \
                          [x for x in self.date_list if valid_end <= x < self.test_start][gap_days:-gap_days]
            train_dates = [x for x in train_dates if x not in self.not_train_date]

            valid_dates = [x for x in self.date_list if valid_start <= x < valid_end][:-gap_days]
            folds.append((train_dates, valid_dates))

        all_train_dates = [x for x in self.date_list if self.valid_date_split[0] <= x < self.test_start][:-gap_days]
        all_train_dates = [x for x in all_train_dates if x not in self.not_train_date]

        test_dates = [d for d in self.date_list
                      if self.test_start <= d <= self.test_end]

        return folds, all_train_dates, test_dates
