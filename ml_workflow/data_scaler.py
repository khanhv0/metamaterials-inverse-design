import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
class DataScaler:
    """Class to handle scaling of both features and targets"""
    def __init__(self, feature_scaler_type='standard', target_scaler_type='standard'):
        self.feature_scaler = self.choose_scaler(feature_scaler_type)
        self.target_scaler = self.choose_scaler(target_scaler_type)
        self.is_fitted = False
    
    def choose_scaler(self, type ='standard'):
        if type == 'minmax':
            return MinMaxScaler() 
        elif type == 'robust':
            return RobustScaler()
        else:
            return StandardScaler()
    
    def fit(self, X, y):
        """Fit scalers on training data"""
        self.X_scaled = self.feature_scaler.fit_transform(X) # Design params
        self.y_scaled = self.target_scaler.fit_transform(y) # Properties 
        self.is_fitted = True
        return self.X_scaled, self.y_scaled
    
    def transform_features(self, X):
        """Transform features using fitted scaler"""
        return self.feature_scaler.transform(X)
    
    def transform_targets(self, y):
        """Transform targets using fitted scaler"""
        return self.target_scaler.transform(y)
    
    # Inverse transform properties
    def inverse_transform_targets(self, y_scaled):
        """Inverse transform targets back to original scale"""
        return self.target_scaler.inverse_transform(y_scaled)
    
    def inverse_transform_features(self, X_scaled):
        return self.feature_scaler.inverse_transform(X_scaled)
    
    def get_scaling_info(self):
        """Get scaling parameters for interpretation"""
        if hasattr(self.target_scaler, 'mean_'):
            return {
                'target_means': self.target_scaler.mean_,
                'target_stds': np.sqrt(self.target_scaler.var_) if hasattr(self.target_scaler, 'var_') else None,
                'target_min': self.target_scaler.min_ if hasattr(self.target_scaler, 'min_') else None,
                'target_max': self.target_scaler.max_ if hasattr(self.target_scaler, 'max_') else None
            }
        return {}
