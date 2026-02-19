"""PCMCI+ baseline for causal inference on temporal data.

PCMCI+ discovers causal graphs from observational data using conditional independence tests,
then uses the discovered graph for causal effect estimation.

Note: This is slow per-sample (~seconds per SCM) since it runs causal discovery each time.
Recommend evaluating on a subset (e.g., 100-500 samples) for computational feasibility.
"""

import numpy as np
from typing import Optional

try:
    from tigramite import data_processing as tigdata
    from tigramite.pcmci import PCMCI
    from tigramite.independence_tests.parcorr import ParCorr
    TIGRAMITE_AVAILABLE = True
except ImportError:
    TIGRAMITE_AVAILABLE = False
    print("Warning: tigramite not available. PCMCI+ baseline will not work.")


class PCMCIBaseline:
    """PCMCI+ baseline for temporal causal inference.
    
    Approach:
    1. Run PCMCI+ on observational time series to discover causal graph
    2. Use discovered parents of target variable for regression adjustment
    3. Predict interventional outcome via adjusted regression
    """
    
    def __init__(self, tau_max: int = 2, alpha_level: float = 0.01):
        """
        Parameters
        ----------
        tau_max : int
            Maximum time lag to consider.
        alpha_level : float
            Significance level for conditional independence tests.
        """
        if not TIGRAMITE_AVAILABLE:
            raise ImportError("tigramite package required for PCMCI+ baseline. "
                            "Install with: pip install tigramite")
        
        self.tau_max = tau_max
        self.alpha_level = alpha_level
        self.graph = None
    
    def fit(self, X_obs: np.ndarray):
        """Run PCMCI+ on observational data to discover causal graph.
        
        Parameters
        ----------
        X_obs : np.ndarray
            Observational time series of shape (T, N).
        """
        T, N = X_obs.shape
        
        # Create tigramite dataframe
        dataframe = tigdata.DataFrame(
            X_obs,
            var_names=[f"X{i}" for i in range(N)]
        )
        
        # Initialize PCMCI with ParCorr (partial correlation) test
        parcorr = ParCorr(significance='analytic')
        pcmci = PCMCI(
            dataframe=dataframe,
            cond_ind_test=parcorr,
            verbosity=0
        )
        
        # Run PCMCI+ algorithm
        # This discovers the time-lagged causal graph
        try:
            results = pcmci.run_pcmciplus(
                tau_min=0,
                tau_max=self.tau_max,
                pc_alpha=self.alpha_level,
            )
            
            # Store the discovered graph
            # results['graph'] has shape (N, N, tau_max+1)
            # results['graph'][i, j, tau] == True means X_j at lag tau causes X_i
            self.graph = results['graph']
            self.val_matrix = results['val_matrix']
            
        except Exception as e:
            # If PCMCI fails (e.g., too few samples), fall back to empty graph
            print(f"PCMCI+ failed: {e}. Using empty graph.")
            self.graph = np.zeros((N, N, self.tau_max + 1), dtype=bool)
            self.val_matrix = np.zeros((N, N, self.tau_max + 1))
    
    def predict_interventional(
        self,
        X_obs: np.ndarray,
        target: int,
        query_target: int,
        query_time: int,
        intervention_target: int,
        intervention_time: int,
        intervention_value: float,
    ) -> float:
        """Predict interventional outcome using discovered graph.
        
        Approach:
        1. Run PCMCI+ on observational data to discover causal graph
        2. For intervened variable: predict using intervention value
        3. For downstream variables: fit regression on discovered parents and predict
        
        Parameters
        ----------
        X_obs : np.ndarray
            Observational time series of shape (T, N).
        target : int
            Deprecated, use query_target instead.
        query_target : int
            Query variable index.
        query_time : int
            Query time index.
        intervention_target : int
            Intervened variable index.
        intervention_time : int
            Time of intervention.
        intervention_value : float
            Intervention value.
            
        Returns
        -------
        float
            Predicted interventional outcome.
        """
        # Run causal discovery on this sample's observational data
        self.fit(X_obs)
        
        T, N = X_obs.shape
        
        # If query is on the intervened variable at or after intervention time
        if query_target == intervention_target and query_time >= intervention_time:
            return intervention_value
        
        # Otherwise, use discovered graph to predict via parent adjustment
        # Get parents of query_target from discovered graph
        parents = []
        if self.graph is not None:
            # graph[j, i, tau] == True means X_i at lag tau causes X_j
            # So parents of query_target are: graph[:, query_target, :]
            for var in range(N):
                for lag in range(self.tau_max + 1):
                    if self.graph[var, query_target, lag]:
                        parents.append((var, lag))
        
        # If no parents found, return observational mean
        if len(parents) == 0:
            return X_obs[:, query_target].mean()
        
        # Fit linear regression on discovered parents
        try:
            from sklearn.linear_model import LinearRegression
            
            # Prepare training data from observational time series
            X_train = []
            y_train = []
            
            for t in range(self.tau_max, T):
                parent_vals = []
                for var, lag in parents:
                    parent_vals.append(X_obs[t - lag, var])
                X_train.append(parent_vals)
                y_train.append(X_obs[t, query_target])
            
            X_train = np.array(X_train)
            y_train = np.array(y_train)
            
            # Fit regression
            reg = LinearRegression()
            reg.fit(X_train, y_train)
            
            # Predict at query_time using intervention-modified parents
            # This is a simplified approach - we modify only the intervened variable
            if query_time >= self.tau_max:
                parent_vals_query = []
                for var, lag in parents:
                    if var == intervention_target and query_time - lag >= intervention_time:
                        # Use intervention value for intervened variable
                        parent_vals_query.append(intervention_value)
                    else:
                        # Use observational value
                        parent_vals_query.append(X_obs[query_time - lag, var])
                
                prediction = reg.predict([parent_vals_query])[0]
                return prediction
            else:
                # If query_time too early, return observational value
                return X_obs[query_time, query_target]
                
        except Exception as e:
            # If regression fails, return observational value
            return X_obs[query_time, query_target] if query_time < T else X_obs[-1, query_target]


def evaluate_pcmci_baseline(
    X_obs_list,
    X_int_list,
    targets_list,
    intervention_times_list,
    intervention_values_list,
    tau_max: int = 2,
    alpha_level: float = 0.01,
):
    """Evaluate PCMCI+ baseline.
    
    Note: This is slow! Each sample requires running PCMCI+ causal discovery.
    
    Parameters
    ----------
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
    tau_max : int
        Maximum lag for PCMCI+.
    alpha_level : float
        Significance level for PCMCI+.
        
    Returns
    -------
    Tuple[float, float]
        (RMSE, MAE)
    """
    if not TIGRAMITE_AVAILABLE:
        print("Warning: tigramite not available. Skipping PCMCI+ baseline.")
        return np.nan, np.nan
    
    baseline = PCMCIBaseline(tau_max=tau_max, alpha_level=alpha_level)
    
    predictions = []
    ground_truths = []
    
    for i in range(len(X_obs_list)):
        X_obs = X_obs_list[i]
        X_int = X_int_list[i]
        target = targets_list[i]
        int_time = intervention_times_list[i]
        int_value = intervention_values_list[i]
        
        # Predict
        try:
            pred = baseline.predict_interventional(X_obs, target, int_time, int_value)
        except Exception as e:
            print(f"PCMCI+ failed on sample {i}: {e}")
            pred = X_obs[int_time, target]  # Fallback to observational value
        
        # Get ground truth
        truth = X_int[int_time, target]
        
        predictions.append(pred)
        ground_truths.append(truth)
        
        # Progress indicator
        if (i + 1) % 10 == 0:
            print(f"   Processed {i + 1}/{len(X_obs_list)} samples...")
    
    predictions = np.array(predictions)
    ground_truths = np.array(ground_truths)
    
    # Compute metrics
    rmse = np.sqrt(np.mean((predictions - ground_truths) ** 2))
    mae = np.mean(np.abs(predictions - ground_truths))
    
    return rmse, mae