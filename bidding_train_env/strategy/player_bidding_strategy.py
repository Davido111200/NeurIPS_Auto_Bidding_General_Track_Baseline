import time
import numpy as np
import os
import psutil
import torch

from bidding_train_env.strategy.base_bidding_strategy import BaseBiddingStrategy
from bidding_train_env.strategy.bc_bidding_strategy import BcBiddingStrategy
from bidding_train_env.strategy.iql_bidding_strategy import IqlBiddingStrategy
from bidding_train_env.strategy.onlinelp_bidding_strategy import OnlineLpBiddingStrategy

class PlayerBiddingStrategy(BaseBiddingStrategy):
    """
    Simple Strategy example for bidding.
    """

    def __init__(self, budget=100, name="PlayerStrategy", cpa=40, category=1):
        """
        Initialize the bidding Strategy.
        parameters:
            @budget: the advertiser's budget for a delivery period.
            @cpa: the CPA constraint of the advertiser.
            @category: the index of advertiser's industry category.

        """
        super().__init__(budget, name, cpa, category)
        self.bc_model = self.return_bc_model()
        self.iql_model = self.return_iql_model()
        # self.onlinelp = self.return_onlinelp_model()

    def return_bc_model(self):
        return BcBiddingStrategy()
    
    def return_iql_model(self):
        return IqlBiddingStrategy()

    def return_onlinelp_model(self):
        return OnlineLpBiddingStrategy()

    def reset(self):
        """
        Reset remaining budget to initial state.
        """
        self.remaining_budget = self.budget

    def bidding(self, timeStepIndex, pValues, pValueSigmas, historyPValueInfo, historyBid,
                historyAuctionResult, historyImpressionResult, historyLeastWinningCost):
        """
        Bids for all the opportunities in a delivery period

        parameters:
         @timeStepIndex: the index of the current decision time step.
         @pValues: the conversion action probability.
         @pValueSigmas: the prediction probability uncertainty.
         @historyPValueInfo: the history predicted value and uncertainty for each opportunity.
         @historyBids: the advertiser's history bids for each opportunity.
         @historyAuctionResult: the history auction results for each opportunity.
         @historyImpressionResult: the history impression result for each opportunity.
         @historyLeastWinningCosts: the history least wining costs for each opportunity.

        return:
            Return the bids for all the opportunities in the delivery period.
        """
        bids_bc = self.bc_model.bidding(timeStepIndex, pValues, pValueSigmas, historyPValueInfo, historyBid,
                historyAuctionResult, historyImpressionResult, historyLeastWinningCost)
        
        bids_iql = self.iql_model.bidding(timeStepIndex, pValues, pValueSigmas, historyPValueInfo, historyBid, 
                historyAuctionResult, historyImpressionResult, historyLeastWinningCost)
        
        # bids_onlinelp = self.onlinelp.bidding(timeStepIndex, pValues, pValueSigmas, historyPValueInfo, historyBid,
        #         historyAuctionResult, historyImpressionResult, historyLeastWinningCost)
        
        bids_ori = self.cpa * pValues

        print("bids_bc: ", bids_bc, "\n"
              "bids_iql: ", bids_iql, "\n"
            #   "bids_onlinelp: ", bids_onlinelp, "\n"
              "bids_ori: ", bids_ori, "\n")
        
        bids = (bids_iql + bids_bc) / 2

        bids = bids_bc
        
        return bids
