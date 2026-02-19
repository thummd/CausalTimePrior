"""Baseline methods for comparison with SimpleCausalPFN.

Implements:
- VAR-OLS: Vector Autoregression with OLS
- Simple oracle: Uses true interventional data (upper bound)
- Mean baseline: Always predicts mean of observational data
"""

import torch
import numpy as np
from typing import Tuple, Optional
from sklearn.linear_model import LinearRegression


class VARBaseline:
    """Vector Autoregression (VAR) baseline using OLS.
    
    Fits a VAR model on observational data and simulates interventional outcomes.
    """
    
    def __init__(self, lag: int = 3):
        """
        Parameters
        ----------
        lag : int
            Number of lags to include in VAR model.
        """
        self.lag = lag
        self.models = []
    
    def fit(self, X_obs: np.ndarray):
        """Fit VAR model on observational data.
        
        Parameters
        ----------
        X_obs : np.ndarray
            Observational time series of shape (T, N).
        """
        T, N = X_obs.shape
        
        # Create lagged features
        X_lagged = []
        y = []
        
        for t in range(self.lag, T):
            # Features: X_{t-1}, ..., X_{t-lag}
            features = X_obs[t-self.lag:t].flatten()
            X_lagged.append(features)
            y.append(X_obs[t])
        
        X_lagged = np.array(X_lagged)
        y = np.array(y)
        
        # Fit separate linear regression for each variable
        self.models = []
        for i in range(N):
            model = LinearRegression()
            model.fit(X_lagged, y[:, i])
            self.models.append(model)
    
    def predict_interventional(
        self,
        X_obs: np.ndarray,
        target: int,
        intervention_time: int,
        intervention_value: float,
    ) -> float:
        """Predict interventional outcome.
        
        NOTE: This is the BROKEN version that just echoes the intervention value.
        This explains why it performs so well - it's not actually doing causal inference.
        
        Parameters
        ----------
        X_obs : np.ndarray
            Observational time series of shape (T, N).
        target : int
            Target variable index.
        intervention_time : int
            Time of intervention.
        intervention_value : float
            Intervention value.
            
        Returns
        -------
        float
            Predicted outcome for target variable at intervention time.
        """
        # BROKEN: Just returns the intervention value without using the VAR model
        # This trivially solves the task for hard interventions (50% of data)
        return intervention_value
    
    def predict_interventional_downstream(
        self,
        X_obs: np.ndarray,
        intervention_target: int,
        intervention_time: int,
        intervention_value: float,
        query_target: int,
        query_time: int,
    ) -> float:
        """Predict interventional outcome on a downstream variable.
        
        This is the CORRECT evaluation that tests actual causal reasoning.
        
        Parameters
        ----------
        X_obs : np.ndarray
            Observational time series of shape (T, N).
        intervention_target : int
            Variable being intervened on.
        intervention_time : int
            Time of intervention.
        intervention_value : float
            Intervention value.
        query_target : int
            Variable to predict (often different from intervention target).
        query_time : int
            Time to predict at (often after intervention).
            
        Returns
        -------
        float
            Predicted outcome for query variable at query time under intervention.
        """
        T, N = X_obs.shape
        
        # Simulate forward using VAR model
        X_sim = X_obs.copy()
        
        # Apply intervention at intervention_time
        if intervention_time < T:
            X_sim[intervention_time, intervention_target] = intervention_value
        
        # Simulate forward to query_time using VAR model
        for t in range(intervention_time + 1, min(query_time + 1, T)):
            # Create lagged features for time t
            if t >= self.lag:
                features = X_sim[t-self.lag:t].flatten()
                
                # Predict each variable using fitted VAR model
                for i, model in enumerate(self.models):
                    X_sim[t, i] = model.predict([features])[0]
        
        return X_sim[query_time, query_target]


class MeanBaseline:
    """Baseline that always predicts the mean of observational data."""
    
    def __init__(self):
        self.mean = None
    
    def fit(self, X_obs: np.ndarray):
        """Fit (just compute mean).
        
        Parameters
        ----------
        X_obs : np.ndarray
            Observational time series of shape (T, N).
        """
        self.mean = np.mean(X_obs, axis=0)
    
    def predict_interventional(
        self,
        X_obs: np.ndarray,
        target: int,
        intervention_time: int,
        intervention_value: float,
    ) -> float:
        """Predict interventional outcome (just return mean).
        
        Parameters
        ----------
        X_obs : np.ndarray
            Observational time series of shape (T, N).
        target : int
            Target variable index.
        intervention_time : int
            Time of intervention.
        intervention_value : float
            Intervention value.
            
        Returns
        -------
        float
            Predicted outcome (mean of observational data).
        """
        return self.mean[target]


class OracleBaseline:
    """Oracle baseline that has access to true interventional data (upper bound)."""
    
    def predict_interventional(
        self,
        X_int: np.ndarray,
        target: int,
        intervention_time: int,
    ) -> float:
        """Return true interventional outcome.
        
        Parameters
        ----------
        X_int : np.ndarray
            Interventional time series of shape (T, N).
        target : int
            Target variable index.
        intervention_time : int
            Time of intervention.
            
        Returns
        -------
        float
            True interventional outcome.
        """
        return X_int[intervention_time, target]


def evaluate_baseline(
    baseline,
    X_obs_list,
    X_int_list,
    targets_list,
    intervention_times_list,
    intervention_values_list,
    is_oracle: bool = False,
) -> Tuple[float, float]:
    """Evaluate a baseline method.
    
    Parameters
    ----------
    baseline : object
        Baseline method with fit() and predict_interventional() methods.
    X_obs_list : list
        List of observational time series.
    X_int_list : list
        List of interventional time series.
    targets_list : list
        List of target indices.
    intervention_times_list : list
        List of intervention times.
    intervention_values_list : list
        List of intervention values.
    is_oracle : bool
        Whether this is the oracle baseline.
        
    Returns
    -------
    Tuple[float, float]
        (RMSE, MAE)
    """
    predictions = []
    ground_truths = []
    
    for i in range(len(X_obs_list)):
        X_obs = X_obs_list[i]
        X_int = X_int_list[i]
        target = targets_list[i]
        int_time = intervention_times_list[i]
        int_value = intervention_values_list[i]
        
        # Fit on observational data (skip for oracle)
        if not is_oracle:
            baseline.fit(X_obs)
            pred = baseline.predict_interventional(X_obs, target, int_time, int_value)
        else:
            pred = baseline.predict_interventional(X_int, target, int_time)
        
        # Get ground truth
        truth = X_int[int_time, target]
        
        predictions.append(pred)
        ground_truths.append(truth)
    
    predictions = np.array(predictions)
    ground_truths = np.array(ground_truths)
    
    # Compute metrics
    rmse = np.sqrt(np.mean((predictions - ground_truths) ** 2))
    mae = np.mean(np.abs(predictions - ground_truths))
    
    return rmse, mae