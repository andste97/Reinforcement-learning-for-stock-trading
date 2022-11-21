# coding=utf-8

"""
Goal: Implement a trading simulator to simulate and compare trading strategies.
Authors: Thibaut Théate and Damien Ernst
Institution: University of Liège
"""

###############################################################################
################################### Imports ###################################
###############################################################################

import os
import sys
import importlib
import pickle
import itertools
import yaml

import numpy as np
import pandas as pd

from tabulate import tabulate
from tqdm import tqdm
from matplotlib import pyplot as plt
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

from tradingEnv import TradingEnv
from tradingPerformance import PerformanceEstimator
from timeSeriesAnalyser import TimeSeriesAnalyser
from TDQN import TDQN



###############################################################################
########################### Class TradingSimulator ############################
###############################################################################

class TradingSimulator:
    """
    GOAL: Accurately simulating multiple trading strategies on different stocks
          to analyze and compare their performance.
        
    VARIABLES: /
          
    METHODS:   - displayTestbench: Display consecutively all the stocks
                                   included in the testbench.
               - analyseTimeSeries: Perform a detailled analysis of the stock
                                    market price time series.
               - plotEntireTrading: Plot the entire trading activity, with both
                                    the training and testing phases rendered on
                                    the same graph.
               - simulateNewStrategy: Simulate a new trading strategy on a 
                                      a certain stock of the testbench.
               - simulateExistingStrategy: Simulate an already existing
                                           trading strategy on a certain
                                           stock of the testbench.
               - evaluateStrategy: Evaluate a trading strategy on the
                                   entire testbench.
               - evaluateStock: Compare different trading strategies
                                on a certain stock of the testbench.
    """

    def __init__(self, run_config_path='./configurations/hyperparameters-default.yml'):
        # 0. SET VARIABLES FROM CONFIG
        with open(run_config_path, 'r') as yamlfile:
            self.run_config = yaml.safe_load(yamlfile)
        environment_params = self.run_config["environment"]

        self.startingDate = environment_params["startingDate"]
        self.endingDate = environment_params["endingDate"]
        self.splittingDate = environment_params["splittingDate"]
        self.actionSpace = environment_params["actionSpace"]
        self.money = environment_params["money"]
        self.stateLength = environment_params["stateLength"]
        self.bounds = environment_params["bounds"]
        self.step = environment_params["step"]
        self.numberOfEpisodes = environment_params["numberOfEpisodes"]
        self.verbose = environment_params["verbose"]
        self.plotTraining = environment_params["plotTraining"]
        self.rendering = environment_params["rendering"]
        self.showPerformance = environment_params["showPerformance"]
        self.saveStrategy = environment_params["saveStrategy"]
        self.fictives = environment_params["fictives"]
        self.strategies = environment_params["strategies"]
        self.stocks = environment_params["stocks"]
        self.indices = environment_params["indices"]
        self.companies = environment_params["companies"]
        self.strategies = environment_params["strategies"]
        self.strategiesAI = environment_params["strategiesAI"]
        self.percentageCosts = environment_params["percentageCosts"]
        self.context = environment_params["context"]

        self.observationSpace = 1 + (self.stateLength - 1) * (4+len(self.context))
        # Variables setting up the default transaction costs
        self.transactionCosts = self.percentageCosts[1] / 100

    def displayTestbench(self):
        """
        GOAL: Display consecutively all the stocks included in the
              testbench (trading indices and companies).
        
        INPUTS: None
        Other inputs specified by configuration:
                - startingDate: Beginning of the trading horizon.
                - endingDate: Ending of the trading horizon.
        
        OUTPUTS: /
        """

        # Display the stocks included in the testbench (trading indices)
        for _, stock in self.indices.items():
            env = TradingEnv(stock, self.startingDate, self.endingDate, 0)
            env.render() 

        # Display the stocks included in the testbench (companies)
        for _, stock in self.companies.items():
            env = TradingEnv(stock, self.startingDate, self.endingDate, 0)
            env.render()


    def analyseTimeSeries(self, stockName):
        """
        GOAL: Perform a detailled analysis of the stock market
              price time series.
        
        INPUTS: - stockName: Name of the stock (in the testbench).
        Other inputs specified by configuration:
                - startingDate: Beginning of the trading horizon.
                - endingDate: Ending of the trading horizon.
                - splitingDate: Spliting date between the training dataset
                                and the testing dataset.
        
        OUTPUTS: /
        """

        # Retrieve the trading stock information
        if(stockName in self.fictives):
            stock = self.fictives[stockName]
        elif(stockName in self.indices):
            stock = self.indices[stockName]
        elif(stockName in self.companies):
            stock = self.companies[stockName]
        # Error message if the stock specified is not valid or not supported
        else:
            print("The stock specified is not valid, only the following stocks are supported:")
            for stock in self.fictives:
                print("".join(['- ', stock]))
            for stock in self.indices:
                print("".join(['- ', stock]))
            for stock in self.companies:
                print("".join(['- ', stock]))
            raise SystemError("Please check the stock specified.")
        
        # TRAINING DATA
        print("\n\n\nAnalysis of the TRAINING phase time series")
        print("------------------------------------------\n")
        trainingEnv = TradingEnv(stock, self.startingDate, self.splittingDate, 0)
        timeSeries = trainingEnv.data['Close']
        analyser = TimeSeriesAnalyser(timeSeries)
        analyser.timeSeriesDecomposition()
        analyser.stationarityAnalysis()
        analyser.cyclicityAnalysis()

        # TESTING DATA
        print("\n\n\nAnalysis of the TESTING phase time series")
        print("------------------------------------------\n")
        testingEnv = TradingEnv(stock, self.splittingDate, self.endingDate, 0)
        timeSeries = testingEnv.data['Close']
        analyser = TimeSeriesAnalyser(timeSeries)
        analyser.timeSeriesDecomposition()
        analyser.stationarityAnalysis()
        analyser.cyclicityAnalysis()

        # ENTIRE TRADING DATA
        print("\n\n\nAnalysis of the entire time series (both training and testing phases)")
        print("---------------------------------------------------------------------\n")
        tradingEnv = TradingEnv(stock, self.startingDate, self.endingDate, 0)
        timeSeries = tradingEnv.data['Close']
        analyser = TimeSeriesAnalyser(timeSeries)
        analyser.timeSeriesDecomposition()
        analyser.stationarityAnalysis()
        analyser.cyclicityAnalysis()


    def plotEntireTrading(self, trainingEnv, testingEnv, strategyName):
        """
        GOAL: Plot the entire trading activity, with both the training
              and testing phases rendered on the same graph for
              comparison purposes.
        
        INPUTS: - trainingEnv: Trading environment for training.
                - testingEnv: Trading environment for testing.
        
        OUTPUTS: /
        """

        # Artificial trick to assert the continuity of the Money curve
        ratio = trainingEnv.data['Money'][-1]/testingEnv.data['Money'][0]
        testingEnv.data['Money'] = ratio * testingEnv.data['Money']

        # Concatenation of the training and testing trading dataframes
        dataframes = [trainingEnv.data, testingEnv.data]
        data = pd.concat(dataframes)

        # Set the Matplotlib figure and subplots
        fig = plt.figure(figsize=(10, 8))
        ax1 = fig.add_subplot(211, ylabel='Price', xlabel='Time')
        ax2 = fig.add_subplot(212, ylabel='Capital', xlabel='Time', sharex=ax1)

        # Plot the first graph -> Evolution of the stock market price
        trainingEnv.data['Close'].plot(ax=ax1, color='blue', lw=2)
        testingEnv.data['Close'].plot(ax=ax1, color='blue', lw=2, label='_nolegend_') 
        ax1.plot(data.loc[data['Action'] == 1.0].index, 
                 data['Close'][data['Action'] == 1.0],
                 '^', markersize=5, color='green')   
        ax1.plot(data.loc[data['Action'] == -1.0].index, 
                 data['Close'][data['Action'] == -1.0],
                 'v', markersize=5, color='red')
        
        # Plot the second graph -> Evolution of the trading capital
        trainingEnv.data['Money'].plot(ax=ax2, color='blue', lw=2)
        testingEnv.data['Money'].plot(ax=ax2, color='blue', lw=2, label='_nolegend_') 
        ax2.plot(data.loc[data['Action'] == 1.0].index, 
                 data['Money'][data['Action'] == 1.0],
                 '^', markersize=5, color='green')   
        ax2.plot(data.loc[data['Action'] == -1.0].index, 
                 data['Money'][data['Action'] == -1.0],
                 'v', markersize=5, color='red')

        # Plot the vertical line seperating the training and testing datasets
        ax1.axvline(pd.Timestamp(self.splittingDate), color='black', linewidth=2.0)
        ax2.axvline(pd.Timestamp(self.splittingDate), color='black', linewidth=2.0)
        
        # Generation of the two legends and plotting
        ax1.legend(["Price", "Long",  "Short", "Train/Test separation"])
        ax2.legend(["Capital", "Long", "Short", "Train/Test separation"])
        figurePath = "Figures/" + strategyName + '_' + trainingEnv.marketSymbol + "_" + trainingEnv.startingDate + "_" + trainingEnv.endingDate + '/'
        os.makedirs(figurePath, exist_ok=True)
        plt.savefig(''.join([figurePath, str(trainingEnv.marketSymbol), '_TrainingTestingRendering', '.png']))
        #plt.show()


    def simulateNewStrategy(self, strategyName, stockName):
        """
        GOAL: Simulate a new trading strategy on a certain stock included in the
              testbench, with both learning and testing phases.
        
        INPUTS: - strategyName: Name of the trading strategy.
                - stockName: Name of the stock (in the testbench).
        Other inputs specified by configuration:
                - startingDate: Beginning of the trading horizon.
                - endingDate: Ending of the trading horizon.
                - splitingDate: Spliting date between the training dataset
                                and the testing dataset.
                - observationSpace: Size of the RL observation space.
                - actionSpace: Size of the RL action space.
                - money: Initial capital at the disposal of the agent.
                - stateLength: Length of the trading agent state.
                - transactionCosts: Additional costs incurred while trading
                                    (e.g. 0.01 <=> 1% of transaction costs).
                - bounds: Bounds of the parameter search space (training).
                - step: Step of the parameter search space (training).
                - numberOfEpisodes: Number of epsiodes of the RL training phase.
                - verbose: Enable the printing of a simulation feedback.
                - plotTraining: Enable the plotting of the training results.
                - rendering: Enable the rendering of the trading environment.
                - showPerformance: Enable the printing of a table summarizing
                                   the trading strategy performance.
                - saveStrategy: Enable the saving of the trading strategy.
        
        OUTPUTS: - tradingStrategy: Trading strategy simulated.
                 - trainingEnv: Trading environment related to the training phase.
                 - testingEnv: Trading environment related to the testing phase.
        """

        # 1. INITIALIZATION PHASE

        # Retrieve the trading strategy information
        if(strategyName in self.strategies):
            strategy = self.strategies[strategyName]
            trainingParameters = [self.bounds, self.step]
            ai = False
        elif(strategyName in self.strategiesAI):
            strategy = self.strategiesAI[strategyName]
            trainingParameters = [self.numberOfEpisodes]
            ai = True
        # Error message if the strategy specified is not valid or not supported
        else:
            print("The strategy specified is not valid, only the following strategies are supported:")
            for strategy in self.strategies:
                print("".join(['- ', strategy]))
            for strategy in self.strategiesAI:
                print("".join(['- ', strategy]))
            raise SystemError("Please check the trading strategy specified.")

        # Retrieve the trading stock information
        if(stockName in self.fictives):
            stock = self.fictives[stockName]
        elif(stockName in self.indices):
            stock = self.indices[stockName]
        elif(stockName in self.companies):
            stock = self.companies[stockName]
        # Error message if the stock specified is not valid or not supported
        else:
            print("The stock specified is not valid, only the following stocks are supported:")
            for stock in self.fictives:
                print("".join(['- ', stock]))
            for stock in self.indices:
                print("".join(['- ', stock]))
            for stock in self.companies:
                print("".join(['- ', stock]))
            raise SystemError("Please check the stock specified.")


        # 2. TRAINING PHASE

        # Initialize the trading environment associated with the training phase

        trainingEnv = TradingEnv(stock, self.startingDate, self.splittingDate, self.money,self.context, self.stateLength, self.transactionCosts)


        # Instanciate the strategy classes
        if ai: #TDQN
            strategyModule = importlib.import_module(str(strategy))
            className = getattr(strategyModule, strategy)
            tradingStrategy = className(self.observationSpace, self.actionSpace, self.run_config)
        else:
            strategyModule = importlib.import_module('classicalStrategy')
            className = getattr(strategyModule, strategy)
            tradingStrategy = className()

        # Training of the trading strategy

        trainingEnv = tradingStrategy.training(trainingEnv, self.context, trainingParameters=trainingParameters,
                                               verbose=self.verbose, rendering=self.rendering,
                                               plotTraining=self.plotTraining, showPerformance=self.showPerformance)

        
        # 3. TESTING PHASE

        # Initialize the trading environment associated with the testing phase

        testingEnv = TradingEnv(stock, self.splittingDate, self.endingDate, self.money, self.context,self.stateLength, self.transactionCosts)

        # Testing of the trading strategy
        testingEnv = tradingStrategy.testing(trainingEnv, testingEnv, rendering=self.rendering, showPerformance=self.showPerformance)
            
        # Show the entire unified rendering of the training and testing phases
        if self.rendering:
            self.plotEntireTrading(trainingEnv, testingEnv, strategyName)


        # 4. TERMINATION PHASE

        # If required, save the trading strategy with Pickle
        if(self.saveStrategy):
            if not os.path.exists('Strategies/'):
                os.makedirs('Strategies/')
            fileName = "".join(["Strategies/", strategy, "_", stock, "_", self.startingDate, "_", self.splittingDate])
            if ai:
                tradingStrategy.saveModel(fileName)
            else:
                fileHandler = open(fileName, 'wb') 
                pickle.dump(tradingStrategy, fileHandler)

        # Return of the trading strategy simulated and of the trading environments backtested
        return tradingStrategy, trainingEnv, testingEnv

    
    def simulateExistingStrategy(self, strategyName, stockName):
        """
        GOAL: Simulate an already existing trading strategy on a certain
              stock of the testbench, the strategy being loaded from the
              strategy dataset. There is no training phase, only a testing
              phase.
        
        INPUTS: - strategyName: Name of the trading strategy.
                - stockName: Name of the stock (in the testbench).
        Other inputs specified by configuration:
                - startingDate: Beginning of the trading horizon.
                - endingDate: Ending of the trading horizon.
                - splitingDate: Spliting date between the training dataset
                                and the testing dataset.
                - observationSpace: Size of the RL observation space.
                - actionSpace: Size of the RL action space.
                - money: Initial capital at the disposal of the agent.
                - stateLength: Length of the trading agent state.
                - transactionCosts: Additional costs incurred while trading
                                    (e.g. 0.01 <=> 1% of transaction costs).
                - rendering: Enable the rendering of the trading environment.
                - showPerformance: Enable the printing of a table summarizing
                                   the trading strategy performance.
        
        OUTPUTS: - tradingStrategy: Trading strategy simulated.
                 - trainingEnv: Trading environment related to the training phase.
                 - testingEnv: Trading environment related to the testing phase.
        """

        # 1. INITIALIZATION PHASE

        # Retrieve the trading strategy information
        if(strategyName in self.strategies):
            strategy = self.strategies[strategyName]
            ai = False
        elif(strategyName in self.strategiesAI):
            strategy = self.strategiesAI[strategyName]
            ai = True
        # Error message if the strategy specified is not valid or not supported
        else:
            print("The strategy specified is not valid, only the following strategies are supported:")
            for strategy in self.strategies:
                print("".join(['- ', strategy]))
            for strategy in self.strategiesAI:
                print("".join(['- ', strategy]))
            raise SystemError("Please check the trading strategy specified.")

        # Retrieve the trading stock information
        if(stockName in self.fictives):
            stock = self.fictives[stockName]
        elif(stockName in self.indices):
            stock = self.indices[stockName]
        elif(stockName in self.companies):
            stock = self.companies[stockName]
        # Error message if the stock specified is not valid or not supported
        else:
            print("The stock specified is not valid, only the following stocks are supported:")
            for stock in self.fictives:
                print("".join(['- ', stock]))
            for stock in self.indices:
                print("".join(['- ', stock]))
            for stock in self.companies:
                print("".join(['- ', stock]))
            raise SystemError("Please check the stock specified.")
        

        # 2. LOADING PHASE    

        # Check that the strategy to load exists in the strategy dataset
        fileName = "".join(["Strategies/", strategy, "_", stock, "_", self.startingDate, "_", self.splittingDate])
        exists = os.path.isfile(fileName)
        # If affirmative, load the trading strategy
        if exists:
            if ai:
                strategyModule = importlib.import_module(strategy)
                className = getattr(strategyModule, strategy)
                tradingStrategy = className(self.observationSpace, self.actionSpace, self.run_config)
                tradingStrategy.loadModel(fileName)
            else:
                fileHandler = open(fileName, 'rb') 
                tradingStrategy = pickle.load(fileHandler)
        else:
            raise SystemError("The trading strategy specified does not exist, please provide a valid one.")


        # 3. TESTING PHASE

        # Initialize the trading environments associated with the testing phase
        trainingEnv = TradingEnv(stock, self.startingDate, self.splittingDate, self.money, self.stateLength, self.transactionCosts)
        testingEnv = TradingEnv(stock, self.splittingDate, self.endingDate, self.money, self.stateLength, self.transactionCosts)

        # Testing of the trading strategy
        trainingEnv = tradingStrategy.testing(trainingEnv, trainingEnv, rendering=self.rendering, showPerformance=self.showPerformance)
        testingEnv = tradingStrategy.testing(trainingEnv, testingEnv, rendering=self.rendering, showPerformance=self.showPerformance)

        # Show the entire unified rendering of the training and testing phases
        if self.rendering:
            self.plotEntireTrading(trainingEnv, testingEnv, strategyName)

        return tradingStrategy, trainingEnv, testingEnv


    def evaluateStrategy(self, strategyName):
        """
        GOAL: Evaluate the performance of a trading strategy on the entire
              testbench of stocks designed.
        
        INPUTS: - strategyName: Name of the trading strategy.
        Other inputs specified by configuration:
                - startingDate: Beginning of the trading horizon.
                - endingDate: Ending of the trading horizon.
                - splitingDate: Spliting date between the training dataset
                                and the testing dataset.
                - observationSpace: Size of the RL observation space.
                - actionSpace: Size of the RL action space.
                - money: Initial capital at the disposal of the agent.
                - stateLength: Length of the trading agent state.
                - transactionCosts: Additional costs incurred while trading
                                    (e.g. 0.01 <=> 1% of transaction costs).
                - bounds: Bounds of the parameter search space (training).
                - step: Step of the parameter search space (training).
                - numberOfEpisodes: Number of epsiodes of the RL training phase.
                - verbose: Enable the printing of simulation feedbacks.
                - plotTraining: Enable the plotting of the training results.
                - rendering: Enable the rendering of the trading environment.
                - showPerformance: Enable the printing of a table summarizing
                                   the trading strategy performance.
                - saveStrategy: Enable the saving of the trading strategy.
        
        OUTPUTS: - performanceTable: Table summarizing the performance of
                                     a trading strategy.
        """
        return self.evaluateStrategyOnStocks(strategyName, self.stocks.keys())



    def evaluateStrategyOnStocks(self, strategyName, stockNames: []):
        """
        GOAL: Evaluate the performance of a trading strategy on the specified stocks.

        INPUTS: - strategyName: Name of the trading strategy.
                - stockNames: List of stock names to test the strategy on
        Other inputs specified by configuration:
                - startingDate: Beginning of the trading horizon.
                - endingDate: Ending of the trading horizon.
                - splitingDate: Spliting date between the training dataset
                                and the testing dataset.
                - observationSpace: Size of the RL observation space.
                - actionSpace: Size of the RL action space.
                - money: Initial capital at the disposal of the agent.
                - stateLength: Length of the trading agent state.
                - transactionCosts: Additional costs incurred while trading
                                    (e.g. 0.01 <=> 1% of transaction costs).
                - bounds: Bounds of the parameter search space (training).
                - step: Step of the parameter search space (training).
                - numberOfEpisodes: Number of epsiodes of the RL training phase.
                - verbose: Enable the printing of simulation feedbacks.
                - plotTraining: Enable the plotting of the training results.
                - rendering: Enable the rendering of the trading environment.
                - showPerformance: Enable the printing of a table summarizing
                                   the trading strategy performance.
                - saveStrategy: Enable the saving of the trading strategy.

        OUTPUTS: - performanceTable: Table summarizing the performance of
                                     a trading strategy.
        """

        # Initialization of some variables
        performanceTable = [["Profit & Loss (P&L)"], ["Annualized Return"], ["Annualized Volatility"], ["Sharpe Ratio"],
                            ["Sortino Ratio"], ["Maximum DrawDown"], ["Maximum DrawDown Duration"], ["Profitability"],
                            ["Ratio Average Profit/Loss"], ["Skewness"]]
        headers = ["Performance Indicator"]

        selected_stocks = {stockName: self.stocks[stockName] for stockName in stockNames}
        # Loop through each stock included in the testbench (progress bar)
        print("Trading strategy evaluation progression:")
        # for stock in tqdm(itertools.chain(indices, companies)):
        for stock in tqdm(selected_stocks):

            # Simulation of the trading strategy on the current stock
            try:
                # Simulate an already existing trading strategy on the current stock
                _, _, testingEnv = self.simulateExistingStrategy(strategyName, stock)
            except SystemError:
                # Simulate a new trading strategy on the current stock
                _, _, testingEnv = self.simulateNewStrategy(strategyName, stock)

            # Retrieve the trading performance associated with the trading strategy
            analyser = PerformanceEstimator(testingEnv.data)
            performance = analyser.computePerformance()

            # Get the required format for the display of the performance table
            headers.append(stock)
            for i in range(len(performanceTable)):
                performanceTable[i].append(performance[i][1])

        # Display the performance table computed
        tabulation = tabulate(performanceTable, headers, tablefmt="fancy_grid", stralign="center")
        print(tabulation)

        # Computation of the average Sharpe Ratio (default performance indicator)
        sharpeRatio = np.mean([float(item) for item in performanceTable[3][1:]])
        print("Average Sharpe Ratio: " + "{0:.3f}".format(sharpeRatio))

        return performanceTable

    def evaluateStock(self, stockName):

        """
        GOAL: Simulate and compare the performance achieved by all the supported
              trading strategies on a certain stock of the testbench.
        
        INPUTS: - stockName: Name of the stock (in the testbench).
        Other inputs specified by configuration:
                - startingDate: Beginning of the trading horizon.
                - endingDate: Ending of the trading horizon.
                - splitingDate: Spliting date between the training dataset
                                and the testing dataset.
                - money: Initial capital at the disposal of the agent.
                - stateLength: Length of the trading agent state.
                - transactionCosts: Additional costs incurred while trading
                                    (e.g. 0.01 <=> 1% of transaction costs).
                - bounds: Bounds of the parameter search space (training).
                - step: Step of the parameter search space (training).
                - numberOfEpisodes: Number of epsiodes of the RL training phase.
                - verbose: Enable the printing of a simulation feedback.
                - plotTraining: Enable the plotting of the training results.
                - rendering: Enable the rendering of the trading environment.
                - showPerformance: Enable the printing of a table summarizing
                                   the trading strategy performance.
                - saveStrategy: Enable the saving of the trading strategy.
        
        OUTPUTS: - performanceTable: Table summarizing the performance of
                                     a trading strategy.
        """

        # Initialization of some variables
        performanceTable = [["Profit & Loss (P&L)"], ["Annualized Return"], ["Annualized Volatility"], ["Sharpe Ratio"], ["Sortino Ratio"], ["Maximum DrawDown"], ["Maximum DrawDown Duration"], ["Profitability"], ["Ratio Average Profit/Loss"], ["Skewness"]]
        headers = ["Performance Indicator"]

        # Loop through all the trading strategies supported (progress bar)
        print("Trading strategies evaluation progression:")
        for strategy in tqdm(itertools.chain(self.strategies, self.strategiesAI)):

            # Simulation of the current trading strategy on the stock
            try:
                # Simulate an already existing trading strategy on the stock
                _, _, testingEnv = self.simulateExistingStrategy(strategy, stockName)
            except SystemError:
                # Simulate a new trading strategy on the stock
                _, _, testingEnv = self.simulateNewStrategy(strategy, stockName)

            # Retrieve the trading performance associated with the trading strategy
            analyser = PerformanceEstimator(testingEnv.data)
            performance = analyser.computePerformance()
            
            # Get the required format for the display of the performance table
            headers.append(strategy)
            for i in range(len(performanceTable)):
                performanceTable[i].append(performance[i][1])

        # Display the performance table
        tabulation = tabulate(performanceTable, headers, tablefmt="fancy_grid", stralign="center")
        print(tabulation)

        return performanceTable
