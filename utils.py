import pandas as pd
import pyarrow
import time


class DataLoader:
    def __init__(self, data_path, fac_name, ret_name, liquid_name):
        self.fac_file = f"{data_path}/factor_data1/{fac_name}/{fac_name}.fea"
        self.ret_file = f"{data_path}/label_data/{ret_name}.fea"
        self.liquid_file = f"{data_path}/label_data/{liquid_name}.fea"
        self.fac_df = None
        self.ret_df = None
        self.liquid_df = None

    def quarter_to_date(self, quarter_str):
        return pd.to_datetime(pd.Period(quarter_str, freq='Q').end_time.date())

    def load_full_data(self):
        # Load factor data
        print(f"Loading factor data from {self.fac_file}...")
        fac_df = pd.read_feather(self.fac_file, columns = None)
        fac_df['date'] = pd.to_datetime(fac_df['date'], format='%Y%m%d')
        self.fac_df = fac_df.set_index(['date','Code'])
        self.fac_df = self.fac_df.astype('float32', copy=False) # Convert to float32 for memory efficiency
        del fac_df
        
        # Load return data
        print(f"Loading return data from {self.ret_file}...")
        ret_df = pd.read_feather(self.ret_file).set_index('index').sort_index()
        ret_df.index = pd.to_datetime(ret_df.index)
        ret_df.index.name = 'date'
        self.ret_df = ret_df
        self.ret_df = self.ret_df.astype('float32', copy=False)  # Convert to float32 for memory efficiency
        del ret_df
       
        # Load liquidity data
        print(f"Loading liquid data from {self.liquid_file}...")
        liquid_df = pd.read_feather(self.liquid_file).set_index('index').sort_index()
        liquid_df.index = pd.to_datetime(liquid_df.index)
        liquid_df.index.name = 'date'
        self.liquid_df = liquid_df
        self.liquid_df = self.liquid_df.astype('float32', copy=False)  # Convert to float32 for memory efficiency
        del liquid_df

    def get_dataset(self, factor_list, test_quarter):
        if self.fac_df is None or self.ret_df is None or self.liquid_df is None:
            raise ValueError("Data not loaded. Please call load_full_data() first.")
        
        test_end_date = self.quarter_to_date(test_quarter)
        test_start_date = test_end_date - pd.DateOffset(months=3)
        train_start_date = test_start_date - pd.DateOffset(years=2)

        # 1. Slice and clean factor data
        factor_list = [c for c in self.fac_df.columns if c in factor_list]
        fac_slice = self.fac_df.loc[(slice(train_start_date, test_end_date), slice(None)), factor_list]
        '''
        nan_ratio = fac_slice.loc[(slice(train_start_date, test_end_date), slice(None)), :].isna().mean()
        valid_factors = nan_ratio[nan_ratio <= 0.3].index.tolist()
        fac_slice = fac_slice[valid_factors]
        '''
        medians = fac_slice.groupby(level='date').median()
        fac_slice = fac_slice.fillna(medians)

        # 2. Slice return and liquidity data
        ret_slice = self.ret_df[train_start_date:test_end_date]
        liquid_slice = self.liquid_df[train_start_date:test_end_date]

        # 3. Wide to long
        ret_long = ret_slice.stack()
        liquid_long = liquid_slice.stack()
        ret_long.index.set_names(["date", "Code"], inplace=True)
        liquid_long.index.set_names(["date", "Code"], inplace=True)
        ret_long.name = "ret"
        liquid_long.name = "liquid"
        
        # 4. Merge
        merged_df = fac_slice.join([ret_long, liquid_long], how='inner')

        return merged_df, fac_slice.columns.tolist()
