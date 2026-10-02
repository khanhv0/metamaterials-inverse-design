from .forward_model import ForwardPropertiesNN
from .inverse_model import InverseDesignNN
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
import torch

class DesignPredictor:
    def __init__(self, forward_path, inverse_path):
        forward_model_state_dict = self.get_field_from_checkpoint('model_state_dict', path=forward_path)
        
        # Load state dict into the forward model
        self.forward_model = ForwardPropertiesNN()
        self.forward_model.load_state_dict(forward_model_state_dict)
        self.forward_model_scaler = self.get_field_from_checkpoint('scaler', path=forward_path)

        inverse_model_state_dict = self.get_field_from_checkpoint('model_state_dict', path=inverse_path)
        
        # Load state dict into the forward model
        self.inverse_model = InverseDesignNN(3, 8)
        self.inverse_model.load_state_dict(inverse_model_state_dict)
        self.inverse_model_scaler = self.get_field_from_checkpoint('scaler', path=inverse_path)


    def create_scaled_dataloader(self, X, batch_size=32):
        X_tensor = torch.FloatTensor(self.inverse_model_scaler.transform_features(X))
        X_dataset = TensorDataset(X_tensor)
        X_loader = DataLoader(X_dataset, batch_size=batch_size, shuffle=False)
        
        return X_loader
    
    
    def predict_from_csv(self, csv_path):
        try:
            df = pd.read_csv(csv_path)
            required_cols = {"young", "poisson", "volume_frac"}
            df_cols = set(df.columns)

            set_diff = required_cols - df_cols
            assert not set_diff, f"The dataframe is missing the following columns: {set_diff}"

        except AssertionError as e:
            print(e)
            return
        except Exception as e:
            print(e)
            return
        
        X = df[["poisson", "young", "volume_frac"]].rename(columns={
            "poisson": " poisson",
            "young": " young",
            "volume_frac": " volume_frac"
        })
        
        loader = self.create_scaled_dataloader(X)
        results = self.predict_designs(loader)
        return results
    

    def get_field_from_checkpoint(self, field, path=None):
        """
        Load checkpoint and populate class attributes.
        Returns the full checkpoint dictionary.
        """

        checkpoint_file = path

        # WARNING: pth needs to be a TRUSTED source, or else weights_only should be True. 
        # Happy unpickling teehee
        checkpoint = torch.load(checkpoint_file, map_location='cpu', weights_only=False)

        # Extract known fields safely
        return checkpoint.get(field, [])
    
    
    def predict_designs(self, loader, device='cpu'):
        self.inverse_model.eval()
        self.forward_model.eval()

        all_target_props_scaled = []
        all_achieved_props_scaled = []
        all_predicted_designs_scaled = []

        with torch.no_grad():
            for batch_X in loader:
                batch_X = batch_X[0].to(device)
                
                # Get predicted designs from inverse model
                predicted_designs_scaled = self.inverse_model(batch_X)
                
                # Convert predicted designs to forward model's input space
                designs_for_forward = self.transform_between_spaces(
                    predicted_designs_scaled,
                    from_scaler=self.inverse_model_scaler, from_is_target=True,
                    to_scaler=self.forward_model_scaler, to_is_target=False
                )
                
                # Get achieved properties from forward model
                achieved_props_scaled = self.forward_model(designs_for_forward)
                
                # Convert achieved properties back to inverse model's input space for loss calculation
                achieved_props_for_inverse = self.transform_between_spaces(
                    achieved_props_scaled,
                    from_scaler=self.forward_model_scaler, from_is_target=True,
                    to_scaler=self.inverse_model_scaler, to_is_target=False
                )
                
                # Store for metrics calculation
                all_target_props_scaled.append(batch_X.cpu().numpy())
                all_achieved_props_scaled.append(achieved_props_for_inverse.cpu().numpy())
                all_predicted_designs_scaled.append(predicted_designs_scaled.cpu().numpy())

        target_props_scaled = np.vstack(all_target_props_scaled)
        achieved_props_scaled = np.vstack(all_achieved_props_scaled)
        predicted_designs_scaled = np.vstack(all_predicted_designs_scaled)
        
        # Inverse transform to original scale for evaluation
        target_props_original = self.inverse_model_scaler.inverse_transform_features(target_props_scaled)
        achieved_props_original = self.inverse_model_scaler.inverse_transform_features(achieved_props_scaled)
        predicted_designs_original = self.inverse_model_scaler.inverse_transform_targets(predicted_designs_scaled)
        problem_rows = self.verify_designs_in_range(predicted_designs_original)
        
        df_data = np.hstack([predicted_designs_original, target_props_original, achieved_props_original, problem_rows])

        results = pd.DataFrame(df_data, columns=[
            "w1", "w2", "w3", "w4", 
            "h4", "d1", "d2", "d3",
            "poisson_requested",
            "young_requested",
            "volume_frac_requested",
            "poisson_predicted",
            "young_predicted",
            "volume_frac_predicted",
            "possibly_problematic"
        ])

        return results

    
    def verify_designs_in_range(self, predicted_designs):
        lower_bound = np.array([1, 0.5, 0.5, 0.25, 7, 0.3, 0.2, 0.2])
        upper_bound = np.array([2.8, 2.5, 3, 1.19, 10.5, 1, 0.85, 1.5])

        problematic = np.any((predicted_designs < lower_bound) | (predicted_designs > upper_bound), axis=1).reshape(-1, 1)
        return problematic


    def transform_between_spaces(self, tensor, from_scaler, from_is_target, to_scaler, to_is_target):
        """
        Transform tensor from one scaled space to another while preserving gradients
        """
        # Step 1: Inverse transform from source space to original units
        if from_is_target:
            from_mean = torch.tensor(from_scaler.target_scaler.mean_, dtype=torch.float32, device=tensor.device)
            from_std = torch.tensor(from_scaler.target_scaler.scale_, dtype=torch.float32, device=tensor.device)
        else:
            from_mean = torch.tensor(from_scaler.feature_scaler.mean_, dtype=torch.float32, device=tensor.device)
            from_std = torch.tensor(from_scaler.feature_scaler.scale_, dtype=torch.float32, device=tensor.device)
        
        original_units = tensor * from_std + from_mean
        
        # Step 2: Transform from original units to target scaled space
        if to_is_target:
            to_mean = torch.tensor(to_scaler.target_scaler.mean_, dtype=torch.float32, device=tensor.device)
            to_std = torch.tensor(to_scaler.target_scaler.scale_, dtype=torch.float32, device=tensor.device)
        else:
            to_mean = torch.tensor(to_scaler.feature_scaler.mean_, dtype=torch.float32, device=tensor.device)
            to_std = torch.tensor(to_scaler.feature_scaler.scale_, dtype=torch.float32, device=tensor.device)
        
        target_scaled = (original_units - to_mean) / to_std
        
        return target_scaled


#predictor = DesignPredictor("./models_pth/forward_model_with_scaler.pth", "./models_pth/inverse_model_with_scaler.pth")
#predictor = DesignPredictor("forward_model_with_scaler.pth", "inverse_model_with_scaler.pth")
#predictor.predict_from_csv("properties_test_samples.csv")