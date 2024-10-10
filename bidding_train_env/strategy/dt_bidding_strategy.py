import numpy as np
import torch
import pickle
import os
from datetime import datetime

from bidding_train_env.strategy.base_bidding_strategy import BaseBiddingStrategy

current_date = datetime.now().strftime("%Y-%m-%d")
current_date = ("2024-08-17")

class DTBiddingStrategy(BaseBiddingStrategy):

    def __init__(self, budget=100, name="Dt-PlayerStrategy", cpa=2, category=1, load_model=True, target_return=50):
        super().__init__(budget, name, cpa, category)


        file_name = os.path.dirname(os.path.realpath(__file__))
        dir_name = os.path.dirname(file_name)
        dir_name = os.path.dirname(dir_name)
        model_path = os.path.join(dir_name, "saved_model", current_date, "DTtest", "dt_model.pth")
        dict_path = os.path.join(dir_name, "saved_model", current_date, "DTtest", "normalize_dict.pkl")
        self.model = None

        if load_model:
            self.model = torch.jit.load(model_path)

            with open(dict_path, 'rb') as f:
                self.normalize_dict = pickle.load(f)
        self.target_return = target_return

    def add_model(self, model):
        if model is not None:
            self.model = model
            self.states = torch.zeros(
                1, self.model.episode_len + 1, self.model.state_dim, dtype=torch.float)
            self.actions = torch.zeros(
                1, self.model.episode_len, self.model.action_dim, dtype=torch.float)
            self.returns = torch.zeros(1, self.model.episode_len + 1, dtype=torch.float)
            self.current_reward = 0
            self.returns[:, 0] = torch.as_tensor(self.target_return)
            self.counter = 0
            self.time_steps = torch.arange(self.model.episode_len, dtype=torch.long)
            self.time_steps = self.time_steps.view(1, -1)

    def reset(self):
        self.remaining_budget = self.budget
        
        self.states = torch.zeros(
            1, self.model.episode_len + 1, self.model.state_dim, dtype=torch.float)  
        self.actions = torch.zeros(
            1, self.model.episode_len, self.model.action_dim, dtype=torch.float)
        self.returns = torch.zeros(1, self.model.episode_len + 1, dtype=torch.float)
        self.current_reward = 0
        self.returns[:, 0] = torch.as_tensor(self.target_return)
        self.counter = 0
        self.time_steps = torch.arange(self.model.episode_len, dtype=torch.long)
        self.time_steps = self.time_steps.view(1, -1)

    def simulate_ad_bidding(self, pValues: np.ndarray,pValueSigmas: np.ndarray, bids: np.ndarray, leastWinningCosts: np.ndarray):
        """
        Simulate the advertising bidding process.

        :param pValues: Values of each pv .
        :param pValueSigmas: uncertainty of each pv .
        :param bids: Bids from the bidding advertiser.
        :param leastWinningCosts: Market prices for each pv.
        :return: Win values, costs spent, and winning status for each bid.

        """
        tick_status = bids >= leastWinningCosts
        tick_cost = leastWinningCosts * tick_status
        values = np.random.normal(loc=pValues, scale=pValueSigmas)
        values = values*tick_status
        tick_value = np.clip(values,0,1)
        tick_conversion = np.random.binomial(n=1, p=tick_value)

        return tick_value, tick_cost, tick_status, tick_conversion

    def bidding(self, timeStepIndex, pValues, pValueSigmas, historyPValueInfo, historyBid,
                historyAuctionResult, historyImpressionResult, historyLeastWinningCost, leastWinningCost):
        """
        Bids for all the opportunities in a delivery period

        parameters:
         @timeStepIndex: the index of the current decision time step.
         @pValues: the conversion action probability.
         @pValueSigmas: the prediction probability uncertainty.
         @historyPValueInfo: the history predicted value and uncertainty for each opportunity.
         @historyBid: the advertiser's history bids for each opportunity.
         @historyAuctionResult: the history auction results for each opportunity.
         @historyImpressionResult: the history impression result for each opportunity.
         @historyLeastWinningCosts: the history least wining costs for each opportunity.

        return:
            Return the bids for all the opportunities in the delivery period.
        """
        time_left = (48 - timeStepIndex) / 48
        budget_left = self.remaining_budget / self.budget if self.budget > 0 else 0
        history_xi = [result[:, 0] for result in historyAuctionResult]
        history_pValue = [result[:, 0] for result in historyPValueInfo]
        history_conversion = [result[:, 1] for result in historyImpressionResult]

        historical_xi_mean = np.mean([np.mean(xi) for xi in history_xi]) if history_xi else 0

        historical_conversion_mean = np.mean(
            [np.mean(reward) for reward in history_conversion]) if history_conversion else 0

        historical_LeastWinningCost_mean = np.mean(
            [np.mean(price) for price in historyLeastWinningCost]) if historyLeastWinningCost else 0

        historical_pValues_mean = np.mean([np.mean(value) for value in history_pValue]) if history_pValue else 0

        historical_bid_mean = np.mean([np.mean(bid) for bid in historyBid]) if historyBid else 0

        def mean_of_last_n_elements(history, n):
            last_three_data = history[max(0, n - 3):n]
            if len(last_three_data) == 0:
                return 0
            else:
                return np.mean([np.mean(data) for data in last_three_data])

        last_three_xi_mean = mean_of_last_n_elements(history_xi, 3)
        last_three_conversion_mean = mean_of_last_n_elements(history_conversion, 3)
        last_three_LeastWinningCost_mean = mean_of_last_n_elements(historyLeastWinningCost, 3)
        last_three_pValues_mean = mean_of_last_n_elements(history_pValue, 3)
        last_three_bid_mean = mean_of_last_n_elements(historyBid, 3)

        current_pValues_mean = np.mean(pValues)
        current_pv_num = len(pValues)

        historical_pv_num_total = sum(len(bids) for bids in historyBid) if historyBid else 0
        last_three_ticks = slice(max(0, timeStepIndex - 3), timeStepIndex)
        last_three_pv_num_total = sum(
            [len(historyBid[i]) for i in range(max(0, timeStepIndex - 3), timeStepIndex)]) if historyBid else 0

        test_state = np.array([
            time_left, budget_left, historical_bid_mean, last_three_bid_mean,
            historical_LeastWinningCost_mean, historical_pValues_mean, historical_conversion_mean,
            historical_xi_mean, last_three_LeastWinningCost_mean, last_three_pValues_mean,
            last_three_conversion_mean, last_three_xi_mean,
            current_pValues_mean, current_pv_num, last_three_pv_num_total,
            historical_pv_num_total
        ])

        def normalize(value, min_value, max_value):
            return (value - min_value) / (max_value - min_value) if max_value > min_value else 0

        for key, value in self.normalize_dict.items():
            test_state[key] = normalize(test_state[key], value["min"], value["max"])

        test_state = torch.tensor(test_state, dtype=torch.float)

        self.states[:, self.counter] = test_state
        self.counter += 1

        with torch.no_grad():
            alpha = self.model(
                self.states[:, : self.counter][:, -self.model.seq_len :],
                self.actions[:, : self.counter][:, -self.model.seq_len :],
                self.returns[:, : self.counter][:, -self.model.seq_len :],
                self.time_steps[:, : self.counter][:, -self.model.seq_len :],
            )

        alpha = alpha[0, -1].cpu().numpy()
        self.actions[:, self.counter - 1] = torch.as_tensor(alpha)
        bids = alpha * pValues

        _, _, _, tick_conversion = self.simulate_ad_bidding(pValues, pValueSigmas, bids, leastWinningCost)
        reward = np.sum(tick_conversion)
        self.returns[:, self.counter] = torch.as_tensor(self.returns[:, self.counter-1] - reward)

        return bids
