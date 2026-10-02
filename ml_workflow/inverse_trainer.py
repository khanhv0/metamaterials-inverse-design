import torch
import warnings
import torch.optim as optim
import numpy as np
import pandas as pd
from typing import override
from forward_model import ForwardPropertiesNN
from data_scaler import DataScaler
from trainer import Trainer
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR, ReduceLROnPlateau


class InverseTrainer(Trainer):
    def __init__(self, forward_model_path, dataset_csv, input_size, output_size, 
                 learning_rate=2e-2, num_epochs = 100,
                 feature_scaler_type='standard', target_scaler_type='standard'):
        # Define attributes the same way as Trainer
        super().__init__(model_type='inverse', dataset_csv=dataset_csv, input_size=input_size, output_size=output_size, 
                 learning_rate=learning_rate, num_epochs=num_epochs,
                 feature_scaler_type=feature_scaler_type, target_scaler_type=target_scaler_type)
        
        # Params related to forward model
        self.forward_model_path = forward_model_path
        self.forward_model = ForwardPropertiesNN()
        self.forward_model_scaler = None
    
    def scale_expected_design_ranges(self):
        """
        Scale the expected ranges from original units to model's scaled space
        """
        # Original expected ranges in original units
        original_ranges = {
            'w1': (0.8, 3),
            'w2': (0.5, 3),
            'w3': (0.5, 3),
            'w4': (0.25, 3),
            'h4': (7, 14),
            'd1': (1, 3),
            'd2': (0.2, 1),
            'd3': (0.2, 1)
        }
        
        # Scale for inverse model's TARGET space (design parameters)
        param_names = ['w1', 'w2', 'w3', 'w4', 'h1', 'd1', 'd2', 'd3']
        original_min = [8.8, 0.5, 0.5, 0.25, 7, 1, 0.2, 0.2]
        original_max = [3, 3, 3, 3, 14, 3, 1, 1]

        bias = 0.1
        original_min = [x + bias for x in original_min]
        original_max = [x + bias for x in original_max]

        original_min = np.array(original_min).reshape(1, -1)  # Reshape to (1, 8)
        original_max = np.array(original_max).reshape(1, -1)  # Reshape to (1, 8)
        warnings.filterwarnings("ignore", message="X does not have valid feature names")

        scaled_min = self.scaler.transform_targets(original_min)
        scaled_max = self.scaler.transform_targets(original_max)
        scaled_min = torch.tensor(scaled_min, dtype=torch.float32).squeeze()  # Convert and squeeze to 1D tensor
        scaled_max = torch.tensor(scaled_max, dtype=torch.float32).squeeze()  # Convert and squeeze to 1D tensor

        return scaled_min, scaled_max

    def apply_design_constraints(self, designs_scaled):
        """
        Apply design constraints in scaled space
        """
        scaled_min, scaled_max = self.scale_expected_design_ranges()
        # Clamp each design parameter to its scaled range
        designs_clamped = torch.clamp(designs_scaled, min=scaled_min, max=scaled_max)
        return designs_clamped
    
    def inverse_transform_tensor(self, tensor):
        """Utility method for inverse transformation that preserves gradients"""
        np_arr = tensor.detach().cpu().numpy()
        np_inv = self.scaler.inverse_transform_targets(np_arr)
        return torch.tensor(np_inv, dtype=torch.float32, device=tensor.device)

    def load_forward_model_and_scaler(self):
        """
        Get forward model and scaler from pth checkpoint file
        """
        # Extract fields 
        forward_model_state_dict = self.get_field_from_checkpoint(field='model_state_dict', path=self.forward_model_path)
        forward_model_scaler = self.get_field_from_checkpoint('scaler', path=self.forward_model_path)
        
        # Load state dict into the forward model
        self.forward_model.load_state_dict(forward_model_state_dict)
        self.forward_model_scaler = forward_model_scaler

    @override
    def train_model_with_scaling(self, model, train_loader, val_loader, num_epochs=100, device='cpu'):
        
        self.load_forward_model_and_scaler()
        model.to(device)
        self.forward_model.to(device)
        
        # Freeze forward model - we don't want to train it, but we need gradients
        for param in self.forward_model.parameters():
            param.requires_grad = False
        
        criterion = torch.nn.MSELoss()  
        self.optimizer = optim.AdamW(model.parameters(), lr=self.learning_rate)
        self.lr_scheduler = ReduceLROnPlateau(self.optimizer, mode='min', factor=0.1, patience=5, 
                                            threshold=0.0001, threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-08)
        
        train_losses = []
        val_losses = []
        
        print("Starting training in SCALED space...")
        
        for epoch in range(num_epochs):
            # Training phase
            model.train()
            self.forward_model.eval()
            train_loss = 0.0
            
            for batch_X, batch_y in train_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                
                # Zero gradients
                self.optimizer.zero_grad()
                
                # Forward pass through inverse model
                # batch_X: scaled target properties → outputs_scaled: scaled design parameters
                outputs_scaled = model(batch_X)
                outputs_clamped= self.apply_design_constraints(outputs_scaled)
                difference = criterion(outputs_scaled, outputs_clamped)
                #print(outputs_scaled.shape[1])
                # Transform inverse model outputs to forward model's input space
                # outputs_scaled is in inverse model's TARGET scaled space
                # We need to convert it to forward model's FEATURE scaled space
                design_params_for_forward = self.transform_between_spaces(
                    outputs_scaled, 
                    from_scaler=self.scaler, from_is_target=True,  # inverse model's target scaler
                    to_scaler=self.forward_model_scaler, to_is_target=False  # forward model's feature scaler
                )
                
                # Forward pass through forward model
                # design_params_for_forward: forward model's scaled features → pred_props_scaled: forward model's scaled targets
                pred_props_scaled = self.forward_model(design_params_for_forward)
                
                # Transform forward model outputs to inverse model's input space
                # pred_props_scaled is in forward model's TARGET scaled space  
                # We need to convert it to inverse model's FEATURE scaled space (to compare with batch_X)
                pred_props_for_inverse = self.transform_between_spaces(
                    pred_props_scaled,
                    from_scaler=self.forward_model_scaler, from_is_target=True,  # forward model's target scaler
                    to_scaler=self.scaler, to_is_target=False  # inverse model's feature scaler
                )
                
                # Calculate loss - both in inverse model's FEATURE scaled space
                #loss = criterion(pred_props_for_inverse, batch_X)
                # TODO: add more weight to young's modulus
                youngs_modulus_weight = 1
                poisson_weight = 1 # 0.9
                volume_fraction_weight = 1
                poisson_loss = criterion(pred_props_for_inverse[:, 0], batch_X[:, 0]) * poisson_weight # Poisson ratio loss
                youngs_modulus_loss = criterion(pred_props_for_inverse[:, 1], batch_X[:, 1]) * youngs_modulus_weight  # Young's modulus with weight
                volume_fraction_loss = criterion(pred_props_for_inverse[:, 2], batch_X[:, 2]) * volume_fraction_weight # Volume fraction loss
                # TODO: add difference to loss
                loss = poisson_loss + youngs_modulus_loss + volume_fraction_loss + 1.23 * difference

                # Backward pass
                loss.backward()
                self.optimizer.step()
                
                train_loss += loss.item() * batch_X.size(0)
            
            train_loss = train_loss / len(train_loader.dataset)
            train_losses.append(train_loss)
            
            # Validation phase
            model.eval()
            val_loss = 0.0
            
            with torch.no_grad():
                for batch_X, batch_y in val_loader:
                    batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                    
                    # Same forward pass as training
                    outputs_scaled = model(batch_X)
                    design_params_for_forward = self.transform_between_spaces(
                        outputs_scaled, 
                        from_scaler=self.scaler, from_is_target=True,
                        to_scaler=self.forward_model_scaler, to_is_target=False
                    )
                    pred_props_scaled = self.forward_model(design_params_for_forward)
                    pred_props_for_inverse = self.transform_between_spaces(
                        pred_props_scaled,
                        from_scaler=self.forward_model_scaler, from_is_target=True,
                        to_scaler=self.scaler, to_is_target=False
                    )
                    
                    outputs_clamped= self.apply_design_constraints(outputs_scaled)
                    
                    loss = criterion(pred_props_for_inverse, batch_X)
                    val_loss += loss.item() * batch_X.size(0)
            
            val_loss = val_loss / len(val_loader.dataset)
            val_losses.append(val_loss)
            
            # TODO: scheduler train loss from val loss
            self.lr_scheduler.step(train_loss)
            
            if (epoch + 1) % 10 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}')
            
            self.current_epoch = epoch
        
        return train_losses, val_losses

    @override
    def test_model_with_scaling(self, model, test_loader, target_columns, scaler, device='cpu'):
        """
        Test the inverse model by checking if predicted designs produce target properties
        when passed through the forward model (the actual training objective)
        """
        model.eval()
        self.forward_model.eval()
        
        # TODO: loss to MAE
        #criterion = torch.nn.MSELoss()
        criterion = torch.nn.L1Loss()
        
        test_loss = 0.0
        all_target_props_scaled = []
        all_achieved_props_scaled = []
        all_predicted_designs_scaled = []
        all_true_designs_scaled = []
        
        print("Testing inverse model with forward model constraint...")
        
        with torch.no_grad():
            for batch_X, batch_y in test_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                
                # Get predicted designs from inverse model
                predicted_designs_scaled = model(batch_X)
                
                # Convert predicted designs to forward model's input space
                designs_for_forward = self.transform_between_spaces(
                    predicted_designs_scaled,
                    from_scaler=self.scaler, from_is_target=True,
                    to_scaler=self.forward_model_scaler, to_is_target=False
                )
                
                # Get achieved properties from forward model
                achieved_props_scaled = self.forward_model(designs_for_forward)
                
                # Convert achieved properties back to inverse model's input space for loss calculation
                achieved_props_for_inverse = self.transform_between_spaces(
                    achieved_props_scaled,
                    from_scaler=self.forward_model_scaler, from_is_target=True,
                    to_scaler=self.scaler, to_is_target=False
                )
                
                # Calculate loss - this matches the training objective
                loss = criterion(achieved_props_for_inverse, batch_X)
                test_loss += loss.item() * batch_X.size(0)
                
                # Store for metrics calculation
                all_target_props_scaled.append(batch_X.cpu().numpy())
                all_achieved_props_scaled.append(achieved_props_for_inverse.cpu().numpy())
                all_predicted_designs_scaled.append(predicted_designs_scaled.cpu().numpy())
                all_true_designs_scaled.append(batch_y.cpu().numpy())
        
        test_loss = test_loss / len(test_loader.dataset)
        
        # Convert to numpy arrays
        target_props_scaled = np.vstack(all_target_props_scaled)
        achieved_props_scaled = np.vstack(all_achieved_props_scaled)
        predicted_designs_scaled = np.vstack(all_predicted_designs_scaled)
        true_designs_scaled = np.vstack(all_true_designs_scaled)
        
        # Inverse transform to original scale for evaluation
        target_props_original = scaler.inverse_transform_features(target_props_scaled)
        achieved_props_original = scaler.inverse_transform_features(achieved_props_scaled)
        predicted_designs_original = scaler.inverse_transform_targets(predicted_designs_scaled)
        true_designs_original = scaler.inverse_transform_targets(true_designs_scaled)
        
        # Calculate metrics on original scale - PROPERTY PREDICTION (training objective)
        print(f"\n=== PROPERTY CONSTRAINT TEST RESULTS (Original Scale) ===")
        print("Measures how well predicted designs achieve target properties via forward model")
        print(f"MSE Loss: {test_loss:.6f}")
        
        # Property prediction metrics
        prop_mae = np.mean(np.abs(achieved_props_original - target_props_original))
        prop_rmse = np.sqrt(np.mean((achieved_props_original - target_props_original) ** 2))
        prop_r_squared = 1 - np.sum((target_props_original - achieved_props_original) ** 2) / np.sum((target_props_original - np.mean(target_props_original)) ** 2)
        
        print(f"Property Prediction RMSE: {prop_rmse:.6f}")
        print(f"Property Prediction MAE: {prop_mae:.6f}")
        print(f"Property Prediction R²: {prop_r_squared:.6f}")
        
        # Calculate metrics for each individual property
        print(f"\n=== Per-Property Metrics (Original Scale) ===")
        
        for i, col_name in enumerate(self.feature_columns):  # Using feature columns for properties
            if i < target_props_original.shape[1]:  # Safety check
                achieved_col = achieved_props_original[:, i]
                target_col = target_props_original[:, i]
                
                mse = np.mean((achieved_col - target_col) ** 2)
                rmse = np.sqrt(mse)
                mae = np.mean(np.abs(achieved_col - target_col))
                mape = np.mean(np.abs((achieved_col - target_col) / np.clip(target_col, 1e-10, None))) * 100
                r_squared = 1 - np.sum((target_col - achieved_col) ** 2) / np.sum((target_col - np.mean(target_col)) ** 2)
                
                print(f"\n{col_name}:")
                print(f"  MSE: {mse:.6f}")
                print(f"  RMSE: {rmse:.6f}")
                print(f"  MAE: {mae:.6f}")
                print(f"  MAPE: {mape:.2f}%")
                print(f"  R²: {r_squared:.6f}")
        
        # Additional: Design prediction accuracy (for reference)
        print(f"\n=== DESIGN PREDICTION ACCURACY (Original Scale) ===")
        print("Measures how close predicted designs are to true designs")
        
        design_mae = np.mean(np.abs(predicted_designs_original - true_designs_original))
        design_rmse = np.sqrt(np.mean((predicted_designs_original - true_designs_original) ** 2))
        design_r_squared = 1 - np.sum((true_designs_original - predicted_designs_original) ** 2) / np.sum((true_designs_original - np.mean(true_designs_original)) ** 2)
        
        print(f"Design Prediction RMSE: {design_rmse:.6f}")
        print(f"Design Prediction MAE: {design_mae:.6f}")
        print(f"Design Prediction R²: {design_r_squared:.6f}")
        
        # Calculate metrics for each individual design parameter
        print(f"\n=== Per-Design Parameter Metrics (Original Scale) ===")
        
        for i, col_name in enumerate(self.target_columns):
            if i < predicted_designs_original.shape[1]:  # Safety check
                pred_col = predicted_designs_original[:, i]
                true_col = true_designs_original[:, i]
                
                mse = np.mean((pred_col - true_col) ** 2)
                rmse = np.sqrt(mse)
                mae = np.mean(np.abs(pred_col - true_col))
                mape = np.mean(np.abs((pred_col - true_col) / np.clip(true_col, 1e-6, None))) * 100
                r_squared = 1 - np.sum((true_col - pred_col) ** 2) / np.sum((true_col - np.mean(true_col)) ** 2)
                
                print(f"\n{col_name}:")
                print(f"  MSE: {mse:.6f}")
                print(f"  RMSE: {rmse:.6f}")
                print(f"  MAE: {mae:.6f}")
                print(f"  MAPE: {mape:.2f}%")
                print(f"  R²: {r_squared:.6f}")
        
        # Store metrics for later use
        self.test_metrics = {
            'property_constraint_loss': test_loss,
            'property_rmse': prop_rmse,
            'property_mae': prop_mae,
            'property_r_squared': prop_r_squared,
            'design_rmse': design_rmse,
            'design_mae': design_mae,
            'design_r_squared': design_r_squared
        }
        
        return test_loss, achieved_props_original, target_props_original
            
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