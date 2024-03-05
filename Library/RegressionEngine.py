from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
from typing import List,Any
from tabulate import tabulate

import numpy as np

import warnings
warnings.filterwarnings("ignore")

class RegressionEngine:
    def __init__(self):
        """
        Initializes the DataPreprocessor class.
        """
        pass
    
    def fitAndPredictAll(self, myAlgorithms: List[Any], X: np.ndarray, Y: np.ndarray, random_state: int) -> None:
        """
        Fits a list of regression algorithms to the training data and evaluates their performance on the test data.

        Parameters:
        - myAlgorithms (List[Any]): A list of instantiated regression algorithms.
        - X (np.ndarray): Feature matrix for training and testing.
        - Y (np.ndarray): Target vector for training and testing.

        Returns:
        - None: Prints a table with the names of algorithms and their R^2, Root Mean Squared Error (RMSE), and Mean Absolute Error (MAE) on the test set.
        """
        x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2, random_state=random_state)
        
        predictions = []

        for algo in myAlgorithms:
            algo_name = type(algo).__name__
            algo.fit(x_train, y_train)
            prediction_result = algo.predict(x_test)

            r_squared = r2_score(y_test, prediction_result)
            rmse = np.sqrt(mean_squared_error(y_test, prediction_result))
            mae = mean_absolute_error(y_test, prediction_result)

            predictions.append([algo_name, r_squared, rmse, mae])

        print(tabulate(predictions, headers=["Algorithm", "R_Squared", "RMSE", "MAE"]))