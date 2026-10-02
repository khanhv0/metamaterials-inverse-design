import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, Subset
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import random
def analyze_dataset(df):
    """Analyze the dataset"""
    print("Dataset Shape:", df.shape)
    print("\nFirst 5 rows:")
    print(df.head())
    
    print("\nDataset Statistics:")
    print(df.describe())
    # correlations between inputs and outputs
    correlation = df.corr()
    
    print(correlation.iloc[:8, 8:])  

class ForwardPropertiesNN(nn.Module):
    def __init__(self, input_size=8, output_size=3):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(64, 32),
            nn.ReLU(),
            
            nn.Linear(32, output_size)
        )
    
    def forward(self, x):
        return self.network(x)

class InverseDesignParamNN1(nn.Module):
    def __init__(self, input_size=3, output_size=8):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, output_size),
            nn.Tanh()
        )
    def forward(self, x):
        return self.network(x)
    
class InverseDesignParamNN2(nn.Module):
    def __init__(self, input_size=3, output_size=8):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, output_size),
            nn.Tanh()
        )
    
    def forward(self, x):
        return self.network(x)
    
class InverseSystem(nn.Module):
    def __init__(self, forward_model, X_scaler, device= None):
        super().__init__()
        self.inverse_model = InverseDesignParamNN2()
        self.forward_model = forward_model
        self.forward_model.eval()
        self.X_scaler = X_scaler
        
        w_1 = [1, 2.8]
        w_2 = [0.5, 2.5]
        w_3 = [0.5, 3]
        w_4 = [0.25, 1.19]
        h_4 = [7, 10.5]
        d_1 = [0.3, 1]
        d_2 = [0.2, 0.85]
        d_3 = [0.2, 1.5]
        
        self.param_ranges = torch.tensor([w_1, w_2, w_3, w_4, h_4, d_1, d_2, d_3])
        if device is None:
            try:
                device = next(forward_model.parameters()).device
            except StopIteration:
                device = torch.device('cpu')
        self.device = device
        self.to(self.device)
        self.param_ranges = self.param_ranges.to(self.device)
    def scale_to_range(self, inverse_output):
        """Scale the output tanh values which is in the range [-1, 1] to actual values"""
        
        min_vals = self.param_ranges[:, 0]
        max_vals = self.param_ranges[:, 1]
        scaled = (inverse_output + 1) / 2
        scaled = scaled * (max_vals - min_vals) + min_vals
        return scaled
    
    def forward(self, target_props):
        """Steps for making an end-to-end predictions"""
        target_props = target_props.to(self.device).float()
        raw_params = self.inverse_model(target_props)
        predicted_params_phys = self.scale_to_range(raw_params)

        mean = torch.from_numpy(self.X_scaler.mean_).to(self.device).float().view(1, -1)
        scale = torch.from_numpy(self.X_scaler.scale_).to(self.device).float().view(1, -1)
        predicted_params_scaled = (predicted_params_phys - mean) / scale
        predicted_props = self.forward_model(predicted_params_scaled)
        
        return predicted_params_scaled, predicted_props
    
def validation_split(dataloader):
    """Split sub dataset into train and validation dataset"""
    # For an internal 87.5/12.5 split for an overall 70/10/20 split
    trainVal_data_size = 0.125 

    if isinstance(dataloader, DataLoader):
        dataset = dataloader.dataset
        batch_size = dataloader.batch_size if dataloader.batch_size is not None else 32
    else:
        dataset = dataloader
        batch_size = 32
    
    indices = list(range(len(dataset)))

    train_idx, val_idx = train_test_split(indices, test_size=trainVal_data_size, shuffle=True)

    train_subset = Subset(dataset, train_idx)
    val_subset = Subset(dataset, val_idx)

    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader

def train_model(model, criterion, optimizer, dataloader, epochs = 10):
    
    train_loader, val_loader = validation_split(dataloader)
    train_losses = []
    val_losses = []

    model.train()
    
    for epoch in range(epochs):
        total_train_loss = 0
        for params, props in train_loader:
            # move data to device
            params= params.to(device)
            props = props.to(device)

            # forward pass
            predictions = model(params)
            loss = criterion(predictions, props)
            
            # backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for params, props in val_loader:
                params, props = params.to(device), props.to(device)
                predictions = model(params)
                loss = criterion(predictions, props)
                total_val_loss += loss.item()

        avg_val_loss = total_val_loss / len(val_loader)
        val_losses.append(avg_val_loss)

        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
    
    # plot the loss curve
    plt.figure(figsize=(8, 5))
    plt.plot(train_losses, label="Training Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training vs Validation Loss")
    plt.legend()
    plt.grid(True)
    plt.show()

def train_inverse_system(inverse_system, dataloader, incriterion, optimizer, epochs = 1000):
    train_loader, val_loader = validation_split(dataloader)
    train_losses = []
    val_losses = []
    param_loss_weight = 0.2
    for epoch in range(epochs):
        total_train_loss = 0
        for props, params in dataloader:
            props = props.to(device).float()
            params = params.to(device).float()
            optimizer.zero_grad()

            # forward pass through system
            predicted_params, predicted_props = inverse_system(props) # TODO: what does this do
            
            # property loss
            prop_loss = criterion(predicted_props, props)
            
            # parameter loss
            param_loss = criterion(predicted_params, params)
            total_loss = prop_loss + param_loss
            
            # backward pass
            # loss in training loop is weighted prop_loss and design param loss
            total_loss = prop_loss + param_loss_weight * param_loss
            total_train_loss += total_loss.item()
            total_loss.backward()
            optimizer.step()
        avg_train_loss = total_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        inverse_system.inverse_model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for props, params in val_loader:
                props, params = props.to(device), params.to(device)
                predictions, _ = inverse_system(props)
                # loss in validation loop is only MSE of design param loss
                loss = criterion(predictions, params)
                total_val_loss += loss.item()

        avg_val_loss = total_val_loss / len(val_loader)
        val_losses.append(avg_val_loss)

        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
    
    # plot the loss curve
    plt.figure(figsize=(8, 5))
    plt.plot(train_losses, label="Training Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training vs Validation Loss")
    plt.legend()
    plt.grid(True)
    plt.show()
        
    return inverse_system

def test_model(model, device, X_test, y_test, X_scaler= None, y_scaler=None):
    model.eval()
    
    # convert test data to tensors
    X_test_tensor = torch.from_numpy(X_test).float().to(device)
    y_test_tensor = torch.from_numpy(y_test).float().to(device)
    
    # make predictions
    with torch.no_grad():
        predictions = model(X_test_tensor)
    
    # convert back to numpy for evaluation
    predictions_np = predictions.cpu().numpy()
    y_true_np = y_test_tensor.cpu().numpy()
    X_test_tensor = X_test_tensor.cpu().numpy()

    if y_scaler is not None:
        predictions_np = y_scaler.inverse_transform(predictions_np)
        y_true_np = y_scaler.inverse_transform(y_true_np)

    if X_scaler is not None:
        X_test_tensor = X_scaler.inverse_transform(X_test_tensor)

    # calculate metrics for each output
    output_names = ['poisson', 'young', 'volume']
    metrics = {}
    print("RESULTS")
    
    for i in range(len(output_names)):
        mse = mean_squared_error(y_true_np[:, i], predictions_np[:, i])
        mae = mean_absolute_error(y_true_np[:, i], predictions_np[:, i])
        r2 = r2_score(y_true_np[:, i], predictions_np[:, i])
        
        metrics[output_names[i]] = {
            'MSE': mse,
            'MAE': mae,
            'R2': r2
        }
        
        print(f"\n{output_names[i].upper()} Prediction:")
        print(f"MSE: {mse:.6f}")
        print(f"MAE: {mae:.6f}")
        print(f"R²: {r2:.4f}")
    
    # overall metrics
    total_mse = mean_squared_error(y_true_np, predictions_np)
    total_mae = mean_absolute_error(y_true_np, predictions_np)
    
    print(f"\n{' OVERALL METRICS ':-^60}")
    print(f"Total MSE:  {total_mse:.6f}")
    print(f"Total MAE:  {total_mae:.6f}")
    indices = random.sample(range(200), 2)
    print("Random outputs")
    for i in indices:
        print(f"\nSample {i+1}:")
        print("Input:", X_test_tensor[i])
        print("Output:", predictions_np[i])
        print("Ground truths:", y_true_np[i])
    return predictions_np, metrics, y_true_np

def test_inverse_model(model, device, X_test, y_test, X_scaler = None, y_scaler=None):
    model.eval()

    # convert test data to tensors
    X_test_tensor = torch.from_numpy(X_test).float().to(device)
    y_test_tensor = torch.from_numpy(y_test).float().to(device)
    
    # make predictions
    with torch.no_grad():
        predictions, _ = model(X_test_tensor)
    
    # convert back to numpy for evaluation
    predictions_np = predictions.cpu().numpy()
    y_true_np = y_test_tensor.cpu().numpy()
    X_test_tensor = X_test_tensor.cpu().numpy()
    if y_scaler is not None:
        predictions_np = y_scaler.inverse_transform(predictions_np)
        y_true_np = y_scaler.inverse_transform(y_true_np)

    if X_scaler is not None:
        X_test_tensor = X_scaler.inverse_transform(X_test_tensor)

    # calculate metrics for each output
    output_names = ['w_1', 'w_2', 'w_3', 'w_4', 'h_4', 'd_1', 'd_2', 'd_3']
    metrics = {}

    print("RESULTS")
    
    for i in range(len(output_names)):
        mse = mean_squared_error(y_true_np[:, i], predictions_np[:, i])
        mae = mean_absolute_error(y_true_np[:, i], predictions_np[:, i])
        r2 = r2_score(y_true_np[:, i], predictions_np[:, i])
        
        metrics[output_names[i]] = {
            'MSE': mse,
            'MAE': mae,
            'R2': r2
        }
        
        print(f"\n{output_names[i].upper()} Prediction:")
        print(f"MSE: {mse:.6f}")
        print(f"MAE: {mae:.6f}")
        print(f"R²: {r2:.4f}")
    
    # overall metrics
    total_mse = mean_squared_error(y_true_np, predictions_np)
    total_mae = mean_absolute_error(y_true_np, predictions_np)
    
    print(f"\n{' OVERALL METRICS ':-^60}")
    print(f"Total MSE:  {total_mse:.6f}")
    print(f"Total MAE:  {total_mae:.6f}")
    indices = random.sample(range(200), 2)
    print("Random outputs")
    for i in indices:
        print(f"\nSample {i+1}:")
        print("Input:", X_test_tensor[i])
        print("Output:", predictions_np[i])
        print("Ground truths:", y_true_np[i])
    return predictions_np, metrics, y_true_np

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    # forward model
    forward_model = ForwardPropertiesNN().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(forward_model.parameters(), lr=0.0005)
    data_file = 'simulation_results.csv'
    training_steps = 15
    test_data_size = 0.2
    # dataset stuff
    #df = generate_synthetic_metamaterial_data(num_samples=5000)
    df = pd.read_csv(data_file)
    X = df.iloc[:, :8].values
    y = df.iloc[:, 8:].values

    # Scale inputs
    X_scaler = StandardScaler()
    X_scaled = X_scaler.fit_transform(X)

    # Scale outputs
    y_scaler = StandardScaler()
    y_scaled = y_scaler.fit_transform(y)

    # Then use X_scaled and y_scaled for training/test split
    
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=test_data_size)
    
    print(f"Training set: {X_train.shape}, {y_train.shape}")
    print(f"Test set: {X_test.shape}, {y_test.shape}")

    X_train_tensor = torch.from_numpy(X_train).float()
    y_train_tensor = torch.from_numpy(y_train).float()
    dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    # test and train forward model
    train_model(forward_model, criterion, optimizer, train_dataloader, training_steps)
    predictions_np, metrics, y_true_np = test_model(forward_model, device, X_test, y_test, X_scaler, y_scaler)
    
    # save model
    forward_file_name = 'forward_NN.pth'
    torch.save(forward_model.state_dict(), forward_file_name)
    print(f"\nForward model saved as {forward_file_name}")

    # Output names
    output_names = ['Poisson', 'Young', 'Volume']
    # Create scatter plots
    plt.figure(figsize=(15, 4))
    for i in range(3):
        plt.subplot(1, 3, i + 1)
        plt.scatter(y_true_np[:, i], predictions_np[:, i], alpha=0.6)
        plt.plot([y_true_np[:, i].min(), y_true_np[:, i].max()],
                [y_true_np[:, i].min(), y_true_np[:, i].max()],
                'r--', lw=2)
        plt.xlabel('True Values')
        plt.ylabel('Predicted Values')
        plt.title(f'{output_names[i]} Prediction')
        plt.grid(True)

    plt.tight_layout()
    plt.show()

    # TODO: add error distribution plots for each parameter
    n_targets = len(output_names)
    fig, axes = plt.subplots(1, n_targets, figsize=(5 * n_targets, 5))
    
    # Ensure axes is iterable
    for i, (ax, col_name) in enumerate(zip(axes, output_names)):
        errors = predictions_np[:, i] - y_true_np[:, i]
        
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
    plt.savefig('error_distributions_forwardNN.png', dpi=300, bbox_inches='tight')
    plt.show()

    # inverse model
    # inverse_system = InverseSystem(forward_model, X_scaler)
    # inverse_system.to(device)
    # inverse_system.inverse_model.train()
    # optimizer_inverse = optim.Adam(inverse_system.inverse_model.parameters(), lr=0.001)
    # criterion_inverse = nn.MSELoss()

    # # dataset stuff
    # X_train, X_test, y_train, y_test = train_test_split(y_scaled, X_scaled, test_size=test_data_size)
    # print(f"Training set: {X_train.shape}, {y_train.shape}")
    # print(f"Test set: {X_test.shape}, {y_test.shape}")
    # X_train_tensor = torch.from_numpy(X_train).float()
    # y_train_tensor = torch.from_numpy(y_train).float()
    # dataset = TensorDataset(X_train_tensor, y_train_tensor)
    # train_dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    # inverse_system = train_inverse_system(inverse_system, train_dataloader, criterion_inverse, optimizer_inverse, 100)
    # test_inverse_model(inverse_system, device, X_test, y_test, y_scaler, X_scaler)

    # # save model
    # inverse_file_name = 'inverse_NN.pth'
    # torch.save(inverse_system.inverse_model.state_dict(), inverse_file_name)
    # print(f"\nForward model saved as {inverse_file_name}")