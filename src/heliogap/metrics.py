import numpy as np

def calculate_wmape(y_true, y_pred):
    """
    Calculate the Weighted Mean Absolute Percentage Error (WMAPE).
    
    Parameters:
        y_true (np.ndarray): Array of true values.
        y_pred (np.ndarray): Array of predicted/interpolated values.
        
    Returns:
        float: The WMAPE score as a percentage.
    """
    error_sum = np.sum(np.abs(y_true - y_pred))
    signal_sum = np.sum(np.abs(y_true))
    
    if signal_sum > 0:
        return (error_sum / signal_sum) * 100
    return 0.0

def calculate_mae(y_true, y_pred):
    """
    Calculate the Mean Absolute Error (MAE).
    
    Parameters:
        y_true (np.ndarray): Array of true values.
        y_pred (np.ndarray): Array of predicted/interpolated values.
        
    Returns:
        float: The MAE score.
    """
    return np.mean(np.abs(y_true - y_pred))

def calculate_rmse(y_true, y_pred):
    """
    Calculate the Root Mean Squared Error (RMSE).
    
    Parameters:
        y_true (np.ndarray): Array of true values.
        y_pred (np.ndarray): Array of predicted/interpolated values.
        
    Returns:
        float: The RMSE score.
    """
    return np.sqrt(np.mean((y_true - y_pred)**2))

def calculate_r2(y_true, y_pred, global_mean):
    """
    Calculate the Coefficient of Determination (R-squared).
    
    Parameters:
        y_true (np.ndarray): Array of true values.
        y_pred (np.ndarray): Array of predicted/interpolated values.
        global_mean (float): The mean of the entire true dataset.
        
    Returns:
        float: The R-squared score.
    """
    ss_res = np.sum((y_true - y_pred)**2)
    ss_tot = np.sum((y_true - global_mean)**2)
    
    if ss_tot > 0:
        return 1 - (ss_res / ss_tot)
    return 0.0

def calculate_mda(y_true_current, y_true_previous, pred_direction):
    """
    Calculate the Mean Directional Accuracy (MDA).
    
    Parameters:
        y_true_current (np.ndarray): Array of true current values.
        y_true_previous (np.ndarray): Array of true previous values.
        pred_direction (int or np.ndarray): Predicted direction sign(s).
        
    Returns:
        float: The MDA score as a percentage.
    """
    true_direction = np.sign(y_true_current - y_true_previous)
    correct_directions = np.sum(true_direction == pred_direction)
    total_points = len(y_true_current)
    
    if total_points > 0:
        return (correct_directions / total_points) * 100
    return 0.0

METRIC_FUNCTIONS = {
    'wmape': calculate_wmape,
    'mae': calculate_mae,
    'rmse': calculate_rmse
}