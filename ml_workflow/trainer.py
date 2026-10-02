import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from forward_model import ForwardPropertiesNN
from inverse_model import InverseDesignNN
from data_scaler import DataScaler
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR, ReduceLROnPlateau


class Trainer:
    def __init__(self, model_type, dataset_csv, input_size, output_size, 
                 learning_rate=2e-2, num_epochs = 100,
                 feature_scaler_type='standard', target_scaler_type='standard'):
        
        # Dataset related params
        self.dataset_csv = dataset_csv
        self.input_size = input_size
        self.output_size = output_size
        self.target_columns = []
        self.feature_columns =[]

        # Model-related params
        self.model_type = model_type
        self.model = None
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.train_losses = []
        self.val_losses = []
        self.target_metrics = dict()
        self.optimzer = None
        self.lr_scheduler = None
        self.current_epoch = 0
        # Scaler
        self.feature_scaler_type = feature_scaler_type
        self.target_scaler_type = target_scaler_type
        self.scaler = DataScaler(feature_scaler_type, target_scaler_type)
    
    # Helper get methods 
    def get_scaler(self):
        return self.scaler
    
    # Preprocessing
    def get_raw_features(self, show_statistics=False):
        df = pd.read_csv(self.dataset_csv)
        if self.model_type =='forward':
            X = df.iloc[:, :self.input_size]  # First 8 columns are features
            y = df.iloc[:, -self.output_size:]  # Last 3 columns are targets
        else:
            y = df.iloc[:, :self.output_size]  # First 8 columns are targets
            X = df.iloc[:, -self.input_size:]  # Last 3 columns are features
        self.target_columns = y.columns.tolist()
        self.feature_columns = X.columns.tolist()
        
        if show_statistics:
            print(f"Target columns: {self.target_columns}")
            print(f"Original target statistics:")
            for col in self.target_columns:
                print(f"  {col}: min={y[col].min():.4f}, max={y[col].max():.4f}, mean={y[col].mean():.4f}, std={y[col].std():.4f}")
            
            print(f"Feature columns: {self.feature_columns}")
            print(f"Original target statistics:")
            for col in self.feature_columns:
                print(f"  {col}: min={y[col].min():.4f}, max={y[col].max():.4f}, mean={y[col].mean():.4f}, std={y[col].std():.4f}")
            
        return X, y
    
    # Scale features, fit Scaler on training data, return scaled features
    def get_scaled_features(self, X_train, X_val, X_test, y_train, y_val, y_test):
        # Fit scaler to training data and return scaled training data
        X_train_scaled, y_train_scaled = self.scaler.fit(X_train, y_train)
        
        # Scale the validation and test samples based on the training data
        X_val_scaled = self.scaler.transform_features(X_val)
        X_test_scaled = self.scaler.transform_features(X_test)
        y_val_scaled = self.scaler.transform_targets(y_val)
        y_test_scaled = self.scaler.transform_targets(y_test)

        print(f"\nScaling applied:")
        print(f"Features: {self.feature_scaler_type}")
        print(f"Targets: {self.target_scaler_type}")
        
        return X_train_scaled, X_val_scaled, X_test_scaled, y_train_scaled, y_val_scaled, y_test_scaled
    
    # Create dataloaders
    def create_data_loaders(self, X_train, X_val, X_test, y_train, y_val, y_test, batch_size=32):
        """
        Create PyTorch DataLoaders for training, validation, and test sets
        """
        # Convert to PyTorch tensors
        X_train_tensor = torch.FloatTensor(X_train)
        X_val_tensor = torch.FloatTensor(X_val)
        X_test_tensor = torch.FloatTensor(X_test)
        y_train_tensor = torch.FloatTensor(y_train)
        y_val_tensor = torch.FloatTensor(y_val)
        y_test_tensor = torch.FloatTensor(y_test)
        
        # Create datasets
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
        test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
        
        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        return train_loader, val_loader, test_loader

    def train_model_with_scaling(self, model, train_loader, val_loader, num_epochs=100, device='cpu'):
        """
        Train the model with scaled data
        """
        model.to(device)
        #criterion = torch.nn.MSELoss()
        criterion = torch.nn.L1Loss()
        self.optimizer = optim.Adam(model.parameters(), lr=self.learning_rate)
        if self.model_type == 'inverse':
            self.optimizer = optim.Adam(model.parameters(), lr=self.learning_rate, weight_decay=1e-4)
        # TODO: add or remove
        #self.lr_scheduler = StepLR(self.optimizer, step_size=20, gamma=0.5) # works ok
        #self.lr_scheduler = CosineAnnealingLR(self.optimizer, T_max=100) # eh
        # TODO: factor changed from 0.1 to 0.5, patience from 5-10
        self.lr_scheduler = ReduceLROnPlateau(self.optimizer, mode='min', factor=0.7, patience=10, threshold=0.0001, threshold_mode='rel', cooldown=2, min_lr=1e-8, eps=1e-08) # works well for forward
        # if self.model_type == 'inverse':
        #     self.lr_scheduler = StepLR(self.optimizer, step_size=20, gamma=0.1)
        # #
        train_losses = []
        val_losses = []
        
        print("Starting training with scaled data...")
        
        for epoch in range(num_epochs):
            # Training phase
            model.train()
            train_loss = 0.0
            
            for batch_X, batch_y in train_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                
                outputs = model(batch_X)
                
                youngs_modulus_weight = 2.3
                poisson_weight = 1.5 # 0.9
                volume_fraction_weight = 1.2
                poisson_loss = criterion(outputs[:, 0], batch_y[:, 0]) * poisson_weight # Poisson ratio loss
                youngs_modulus_loss = criterion(outputs[:, 1], batch_y[:, 1]) * youngs_modulus_weight  # Young's modulus with weight
                volume_fraction_loss = criterion(outputs[:, 2], batch_y[:, 2]) * volume_fraction_weight # Volume fraction loss
                loss = poisson_loss + youngs_modulus_loss + volume_fraction_loss
                #loss = criterion(outputs, batch_y)

                self.optimizer.zero_grad()
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
                    outputs = model(batch_X)

                    youngs_modulus_weight = 2.3
                    poisson_weight = 1.5 # 0.9
                    volume_fraction_weight = 1.2
                    poisson_loss = criterion(outputs[:, 0], batch_y[:, 0]) * poisson_weight # Poisson ratio loss
                    youngs_modulus_loss = criterion(outputs[:, 1], batch_y[:, 1]) * youngs_modulus_weight  # Young's modulus with weight
                    volume_fraction_loss = criterion(outputs[:, 2], batch_y[:, 2]) * volume_fraction_weight # Volume fraction loss
                    loss = poisson_loss + youngs_modulus_loss + volume_fraction_loss
                    #loss = criterion(outputs, batch_y)
                    
                    val_loss += loss.item() * batch_X.size(0)
            
            val_loss = val_loss / len(val_loader.dataset)
            val_losses.append(val_loss)
            
            # TODO: remove or not remove val_loss
            if self.model_type == 'forward':
                self.lr_scheduler.step(val_loss)
            if (epoch + 1) % 10 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}')
            
            self.current_epoch = epoch
        return train_losses, val_losses
    
    
    def test_model_with_scaling(self, model, test_loader, target_columns, scaler, device='cpu'):
        """
        Test the model and inverse transform predictions to original scale
        """
        model.eval()
        #criterion = torch.nn.MSELoss()
        criterion = torch.nn.L1Loss()
        
        test_loss = 0.0
        all_predictions_scaled = []
        all_targets_scaled = []
        
        with torch.no_grad():
            for batch_X, batch_y in test_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                test_loss += loss.item() * batch_X.size(0)
                
                all_predictions_scaled.append(outputs.cpu().numpy())
                all_targets_scaled.append(batch_y.cpu().numpy())
        
        test_loss = test_loss / len(test_loader.dataset)
        
        # Convert to numpy arrays (still scaled)
        predictions_scaled = np.vstack(all_predictions_scaled)
        targets_scaled = np.vstack(all_targets_scaled)
        
        # Inverse transform to original scale for evaluation
        predictions_original = scaler.inverse_transform_targets(predictions_scaled)
        targets_original = scaler.inverse_transform_targets(targets_scaled)
        
        # Calculate metrics on original scale
        mae = np.mean(np.abs(predictions_original - targets_original))
        rmse = np.sqrt(np.mean((predictions_original - targets_original) ** 2))
        
        print(f"\n=== Test Results (Original Scale) ===")
        print(f"MSE Loss: {test_loss:.6f}")
        print(f"RMSE: {rmse:.6f}")
        print(f"MAE: {mae:.6f}")
        
        # Calculate metrics for each individual target
        print(f"\n=== Per-Target Metrics (Original Scale) ===")
        
        for i, col_name in enumerate(self.target_columns):
            pred_col = predictions_original[:, i]
            target_col = targets_original[:, i]
            
            mse = np.mean((pred_col - target_col) ** 2)
            rmse = np.sqrt(mse)
            mae = np.mean(np.abs(pred_col - target_col))
            mape = np.mean(np.abs((pred_col - target_col) / target_col)) * 100
            r_squared = 1 - np.sum((target_col - pred_col) ** 2) / np.sum((target_col - np.mean(target_col)) ** 2)
            
            self.target_metrics[col_name] = {
                'MSE': mse,
                'RMSE': rmse,
                'MAE': mae,
                'MAPE': mape,
                'R²': r_squared
            }
            
            print(f"\n{col_name}:")
            print(f"  MSE: {mse:.6f}")
            print(f"  RMSE: {rmse:.6f}")
            print(f"  MAE: {mae:.6f}")
            print(f"  MAPE: {mape:.2f}%")
            print(f"  R²: {r_squared:.6f}")
        
        return test_loss, predictions_original, targets_original

    def train(self):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {device}")

        # Get raw features
        X, y = self.get_raw_features()

        # Train–Val–Test split
        X_train_val, X_test, y_train_val, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_val, y_train_val, test_size=0.125, random_state=42
        )

        # Apply scaling
        X_train_scaled, X_val_scaled, X_test_scaled, y_train_scaled, y_val_scaled, y_test_scaled = self.get_scaled_features(X_train, X_val, X_test, y_train, y_val, y_test)
        
        # Create data loaders with scaled data
        train_loader, val_loader, test_loader = self.create_data_loaders(
            X_train_scaled, X_val_scaled, X_test_scaled, 
            y_train_scaled, y_val_scaled, y_test_scaled, 
        )

        # Initialize model
        if self.model_type == "forward":
            self.model = ForwardPropertiesNN(input_size=self.input_size, output_size=self.output_size)
        else:
            self.model = InverseDesignNN(input_size=self.input_size, output_size=self.output_size)
        
        print(f"\n {self.model_type.capitalize()} model was initialized with input_size={self.input_size}, output_size={self.output_size}")

        # Train the model
        train_losses, val_losses = self.train_model_with_scaling(
            model=self.model,
            train_loader=train_loader,
            val_loader=val_loader,
            num_epochs=self.num_epochs,
            device ='cpu'
        )
        self.train_losses = train_losses
        print(f"Train_losses check: {train_losses[0]}")
        self.val_losses = val_losses
        # Plot losses
        self.plot_losses(train_losses, val_losses)
        
        # Test model - remember to inverse transform predictions!
        test_loss, predictions_scaled, targets_scaled = self.test_model_with_scaling(
            self.model, test_loader, self.target_columns, self.scaler, device
        )
        
        # Plot results
        if self.model_type == 'forward':
            plot_col_names = self.target_columns
        else:
            plot_col_names = self.feature_columns
        self.plot_predictions_vs_true(predictions_scaled, targets_scaled, plot_col_names)
        self.plot_error_distribution(predictions_scaled, targets_scaled, plot_col_names)
        
        # Save model and scaler
        self.save_model()

    def save_model(self):
        """
        Save both model and scaler for future use
        """
        filepath = f'{self.model_type}_model_with_scaler.pth'
        torch.save({
            'model_type':self.model_type,
            'model_state_dict': self.model.state_dict(),
            'feature_scaler': self.scaler.feature_scaler,
            'target_scaler': self.scaler.target_scaler,
            'scaler': self.scaler,
            'feature_scaler_type': self.feature_scaler_type,
            'target_scaler_type': self.target_scaler_type,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'lr_scheduler_state_dict': self.lr_scheduler.state_dict(),
            'epoch': self.current_epoch,

        }, filepath)
        print(f"Model and scaler saved to {filepath}")    
    
    def plot_losses(self, train_losses, val_losses):
        """
        Plot training and validation losses
        """
        plt.figure(figsize=(10, 6))
        plt.plot(train_losses, label='Training Loss')
        plt.plot(val_losses, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Losses')
        plt.legend()
        plt.grid(True)
        plt.savefig(f'{self.model_type}_training_losses.png', dpi=300, bbox_inches='tight')
        plt.show()

    def plot_predictions_vs_true(self, predictions, targets, target_columns):
        """
        Plot predictions vs true values in original scale
        """
        n_targets = len(target_columns)
        
        fig, axes = plt.subplots(1, n_targets, figsize=(5*n_targets, 5))
        
        if n_targets == 1:
            axes = [axes]
        
        for i, (ax, col_name) in enumerate(zip(axes, target_columns)):
            pred_col = predictions[:, i]
            target_col = targets[:, i]
            
            ax.scatter(target_col, pred_col, alpha=0.6, s=20)
            
            min_val = min(target_col.min(), pred_col.min())
            max_val = max(target_col.max(), pred_col.max())
            ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect prediction')
            
            r_squared = 1 - np.sum((target_col - pred_col) ** 2) / np.sum((target_col - np.mean(target_col)) ** 2)
            
            ax.set_xlabel(f'True {col_name}')
            ax.set_ylabel(f'Predicted {col_name}')
            ax.set_title(f'{col_name}\nR² = {r_squared:.4f}')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{self.model_type}_predictions_vs_true_scaled.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_error_distribution(self, predictions, targets, target_columns):
        """
        Plot error distribution for each target in original scale
        """
        n_targets = len(target_columns)
        
        fig, axes = plt.subplots(1, n_targets, figsize=(5*n_targets, 5))
        
        if n_targets == 1:
            axes = [axes]
        
        for i, (ax, col_name) in enumerate(zip(axes, target_columns)):
            errors = predictions[:, i] - targets[:, i]
            
            # Compute metrics
            mean_err = np.mean(errors)
            std_err = np.std(errors)
            var_err = np.var(errors)
            min_err = np.min(errors)
            max_err = np.max(errors)
            
            # Plot histogram of errors
            ax.hist(errors, bins=30, alpha=0.7, edgecolor='black')
            ax.axvline(x=0, color='r', linestyle='--', label='Zero error')
            ax.axvline(x=np.mean(errors), color='g', linestyle='-', label=f'Mean: {np.mean(errors):.4f}')
            
            ax.set_xlabel(f'Error ({col_name})')
            ax.set_ylabel('Frequency')
            ax.set_title(f'Error Distribution - {col_name}')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            ax.text(
                0.5, -0.18,  # Position relative to the subplot
                f"mean = {mean_err:.4f} | std = {std_err:.4f}| var= {var_err:.4f} | min= {min_err:.4f} | max= {max_err:.4f}",
                ha='center', va='top', transform=ax.transAxes, fontsize=9, color='gray'
            )
        
        plt.tight_layout()
        plt.savefig(f'{self.model_type}_model_error_distributions.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def get_field_from_checkpoint(self, field, path=None):
        """
        Load checkpoint and populate class attributes.
        Returns the full checkpoint dictionary.
        """
        
        checkpoint_file = path
        if checkpoint_file is None:
            checkpoint_file = f'{self.model_type}_model_with_scaler.pth'

        # WARNING: pth needs to be a TRUSTED source, or else weights_only should be True. 
        # Happy unpickling teehee
        checkpoint = torch.load(checkpoint_file, map_location='cpu', weights_only=False)

        # Extract known fields safely
        return checkpoint.get(field, [])
    
    def load_field_from_checkpoint(self, field, path=None, warn=True):
        # Determine checkpoint path
        checkpoint_file = path if path is not None else f'{self.model_type}_model_with_scaler.pth'

        if checkpoint_file is None:
            raise ValueError("No checkpoint path provided and `self.checkpoint_path` is not set.")

        # Load checkpoint
        try:
            checkpoint = torch.load(checkpoint_file, map_location="cpu", weights_only=False)
        except FileNotFoundError:
            raise FileNotFoundError(f"Checkpoint file not found at: {path}")

        # Missing field, skip
        if field not in checkpoint:
            if warn:
                print(f"[WARN] Field '{field}' not found in checkpoint. Skipping.")
            return None

        value = checkpoint[field]

        # Check if the class has the matching attribute
        if not hasattr(self, field):
            if warn:
                print(f"[WARN] Class has no attribute '{field}'. Skipping assignment.")
            return False

        # Assign value to class attribute
        setattr(self, field, value)

        if warn:
            print(f"[INFO] Loaded '{field}' from checkpoint into class attribute '{field}'.")

        return True