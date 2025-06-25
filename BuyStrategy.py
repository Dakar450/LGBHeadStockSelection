import pandas as pd
import numpy as np
from typing import Literal


class BuyStrategy:
    def __init__(
        self,
        df: pd.DataFrame,
        top_pct: float,
        budget: float,
        strategy_type: Literal["greedy", "equal", "liquid_equal"] = "greedy"
    ):
        self.df = df.copy()
        self.pred_col = "pred"
        self.ret_col = "ret"
        self.liquid_col = "liquid"
        self.top_pct = top_pct
        self.budget = budget
        self.strategy_type = strategy_type

    def _greedy_buy(self, selected: pd.DataFrame) -> pd.DataFrame:
        selected.sort_values(by=self.pred_col, ascending=False, inplace=True)
        budget_left = self.budget
        buy_amt = []
        for l in selected[self.liquid_col]:
            amt = min(budget_left, l)
            buy_amt.append(amt)
            budget_left -= amt
        selected["buy_amt"] = buy_amt
        return selected

    def _equal_buy(self, selected: pd.DataFrame) -> pd.DataFrame:
        n = len(selected)
        liq = selected[self.liquid_col].values.copy()
        buy_amt = np.zeros(n)
        budget_left = self.budget

        while budget_left > 1e-6:
            remain_room = liq - buy_amt
            active = remain_room > 1e-6
            n_active = np.sum(active)
            if n_active == 0:
                break
            equal_amt = budget_left / n_active
            delta = np.minimum(equal_amt, remain_room)
            buy_amt += delta
            budget_left -= delta.sum()

        selected["buy_amt"] = buy_amt
        return selected

    def _liquid_equal_buy(self, selected: pd.DataFrame) -> pd.DataFrame:
        liq = selected[self.liquid_col].values.copy()
        buy_amt = np.zeros(len(selected))
        budget_left = self.budget

        while budget_left > 1e-6:
            remain_room = liq - buy_amt
            active = remain_room > 1e-6
            remain_liq = liq[active]
            if remain_liq.sum() == 0:
                break
            alloc_ratio = remain_liq / remain_liq.sum()
            delta = np.zeros_like(buy_amt)
            delta[active] = np.minimum(budget_left * alloc_ratio, remain_room[active])
            buy_amt += delta
            budget_left -= delta.sum()

        selected["buy_amt"] = buy_amt
        return selected

    def run(self):
        results = []
        for date, group in self.df.groupby(level = "date"):
            group = group.copy()
            n_select = max(1, int(len(group) * self.top_pct))
            selected = group.nlargest(n_select, self.pred_col)

            if selected[self.liquid_col].sum() <= self.budget:
                selected["buy_amt"] = selected[self.liquid_col]
            else:
                if self.strategy_type == "greedy":
                    selected = self._greedy_buy(selected)
                elif self.strategy_type == "equal":
                    selected = self._equal_buy(selected)
                elif self.strategy_type == "liquid_equal":
                    selected = self._liquid_equal_buy(selected)
                else:
                    raise ValueError("Invalid strategy_type")

            selected["weight"] = selected["buy_amt"] / self.budget
            selected["daily_ret"] = selected["weight"] * selected[self.ret_col]

            results.append({
                "date": date,
                "weighted_return": selected["daily_ret"].sum(),
                "position_ratio": selected["buy_amt"].sum() / self.budget
            })

        return pd.DataFrame(results)
