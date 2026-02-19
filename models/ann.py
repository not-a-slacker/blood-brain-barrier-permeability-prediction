# %% [markdown]
# # Blood-Brain Barrier Permeability Prediction using Artificial Neural Networks
# 
# This notebook implements an ANN model for predicting blood-brain barrier (BBB) permeability with:
# - **Hyperparameter Tuning**: Number of layers, neurons per layer, learning rate, batch size
# - **Wandb Integration**: For visualisation of results and comparison of different models
# - **Feature Engineering**: Molecular descriptors + MACCS fingerprints as input to model
# - **Dataset**: B3DB dataset
# 
# ## Table of Contents
# 1. Import Libraries
# 2. Load and Explore Data
# 3. Feature Extraction
# 4. Data Preprocessing
# 5. ANN Model Architecture
# 6. Hyperparameter Tuning with Wandb
# 7. Model Training
# 8. Evaluation and Results

# %% [markdown]
# ## 1. Import Required Libraries

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import time
import warnings
warnings.filterwarnings('ignore')

from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski, Crippen, MACCSkeys

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset

from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, 
                             classification_report, confusion_matrix, roc_curve, auc,
                             roc_auc_score, matthews_corrcoef)

# Wandb for experiment tracking
import wandb

np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


# %% [markdown]
# ## 2. Load and Explore Data

# %%
data_path = Path('../data/B3DB_classification.tsv')
df = pd.read_csv(data_path, sep='\t')

print(f"Dataset shape: {df.shape}")
print(f"\nFirst few rows:")
print(df.head())

print(f"\nDataset info:")
print(df.info())

print(f"\nTarget distribution:")
print(df['BBB+/BBB-'].value_counts())
print(f"\nClass balance:")
print(df['BBB+/BBB-'].value_counts(normalize=True))

print(f"\nMissing values:")
print(df.isnull().sum())

# %% [markdown]
# ## 3. Feature Extraction
# 
# We'll extract two types of features:
# 1. **Molecular Descriptors**: Physical and chemical properties (12 descriptors)
# 2. **MACCS Fingerprints**: 167-bit structural fingerprints
# 
# Total feature dimension: 12 + 167 = 179 features

# %%
def extract_molecular_descriptors(smiles):
    """Extract molecular descriptors from SMILES string"""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    
    descriptors = {
        'MolWt': Descriptors.MolWt(mol),
        'LogP': Crippen.MolLogP(mol),
        'NumHDonors': Lipinski.NumHDonors(mol),
        'NumHAcceptors': Lipinski.NumHAcceptors(mol),
        'TPSA': Descriptors.TPSA(mol),
        'NumRotatableBonds': Lipinski.NumRotatableBonds(mol),
        'NumAromaticRings': Lipinski.NumAromaticRings(mol),
        'NumHeteroatoms': Lipinski.NumHeteroatoms(mol),
        'NumRings': Lipinski.RingCount(mol),
        'FractionCsp3': Lipinski.FractionCSP3(mol),
        'NumSaturatedRings': Lipinski.NumSaturatedRings(mol),
        'NumAliphaticRings': Lipinski.NumAliphaticRings(mol),
    }
    return descriptors

def extract_maccs_fingerprint(smiles):
    """Extract MACCS fingerprint from SMILES string"""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    
    maccs = MACCSkeys.GenMACCSKeys(mol)
    return np.array(maccs)

def extract_features(smiles):
    """Extract combined features: molecular descriptors + MACCS fingerprints"""
    descriptors = extract_molecular_descriptors(smiles)
    if descriptors is None:
        return None
    
    maccs = extract_maccs_fingerprint(smiles)
    if maccs is None:
        return None
    
    # Combine features
    descriptor_values = list(descriptors.values())
    combined_features = np.concatenate([descriptor_values, maccs])
    
    return combined_features


# %%
features_list = []
valid_indices = []

for idx, smiles in enumerate(df['SMILES']):
    features = extract_features(smiles)
    if features is not None:
        features_list.append(features)
        valid_indices.append(idx)
    
    if (idx + 1) % 500 == 0:
        print(f"Processed {idx + 1}/{len(df)} molecules...")

X = np.array(features_list)
y = df.loc[valid_indices, 'BBB+/BBB-'].values

print(f"Total samples: {len(X)}")
print(f"Feature dimension: {X.shape[1]}")
print(f"Valid molecules: {len(valid_indices)}/{len(df)}")
print(f"\nFeature matrix shape: {X.shape}")
print(f"Target vector shape: {y.shape}")

# %% [markdown]
# ## 4. Data Preprocessing
# 
# Split the data into training and test sets, then standardize features. [We use stratified sampling]

# %%
# Split data into train and test sets (80-20 split)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Training set size: {len(X_train)}")
print(f"Test set size: {len(X_test)}")
print(f"\nTraining set class distribution:")
print(pd.Series(y_train).value_counts(normalize=True))
print(f"\nTest set class distribution:")
print(pd.Series(y_test).value_counts(normalize=True))

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"Scaled training data shape: {X_train_scaled.shape}")
print(f"Scaled test data shape: {X_test_scaled.shape}")

# %% [markdown]
# ## 5. ANN Model Architecture
# 
# Define a flexible neural network architecture that supports variable number of layers and neurons.

# %%
class BBBPredictorANN(nn.Module):
    """
    ANN architecture for BBB permeability prediction
    
    Parameters:
    -----------
    input_size : int
        Dimension of input features
    hidden_layers : list of int
        Number of neurons in each hidden layer
    dropout_rate : float
        Dropout rate for regularization
    """
    def __init__(self, input_size, hidden_layers, dropout_rate=0.3):
        super(BBBPredictorANN, self).__init__()
        
        layers = []
        prev_size = input_size
        
        for hidden_size in hidden_layers:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm1d(hidden_size))
            layers.append(nn.Dropout(dropout_rate))
            prev_size = hidden_size
        
        layers.append(nn.Linear(prev_size, 1))
        layers.append(nn.Sigmoid())
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)

# %%
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, device):
    """Train the ANN model and track performance"""
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    
    best_val_acc = 0.0
    best_model_state = None
    
    for epoch in range(num_epochs):
        #training
        start_time = time.time()
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            
            # Forward pass
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y.unsqueeze(1))
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Calculate accuracy
            predictions = (outputs > 0.5).float()
            train_correct += (predictions == batch_y.unsqueeze(1)).sum().item()
            train_total += batch_y.size(0)
            train_loss += loss.item()
        
        avg_train_loss = train_loss / len(train_loader)
        train_acc = train_correct / train_total
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y.unsqueeze(1))
                
                predictions = (outputs > 0.5).float()
                val_correct += (predictions == batch_y.unsqueeze(1)).sum().item()
                val_total += batch_y.size(0)
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(val_loader)
        val_acc = val_correct / val_total if val_total > 0 else 0
        
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)
        
        wandb.log({
            'epoch': epoch + 1,
            'train_loss': avg_train_loss,
            'val_loss': avg_val_loss,
            'train_accuracy': train_acc,
            'val_accuracy': val_acc
        })
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict().copy()
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}] - "
                  f"Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.4f} - "
                  f"Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.4f}, "
                  f"Time for epoch: {time.time() - start_time:.2f} seconds")
    
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    return model, {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'train_accs': train_accs,
        'val_accs': val_accs,
        'best_val_acc': best_val_acc
    }

def evaluate_model(model, test_loader, device):
    """Evaluate the model on test set"""
    model.eval()
    all_predictions = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            
            outputs = model(batch_X)
            predictions = (outputs > 0.5).float()
            
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(batch_y.cpu().numpy())
            all_probs.extend(outputs.cpu().numpy())
    
    all_predictions = np.array(all_predictions).flatten()
    all_labels = np.array(all_labels).flatten()
    all_probs = np.array(all_probs).flatten()
    
    accuracy = accuracy_score(all_labels, all_predictions)
    precision = precision_score(all_labels, all_predictions)
    recall = recall_score(all_labels, all_predictions)
    f1 = f1_score(all_labels, all_predictions)
    roc_auc = roc_auc_score(all_labels, all_probs)
    mcc = matthews_corrcoef(all_labels, all_predictions)
    
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'roc_auc': roc_auc,
        'mcc': mcc
    }
    
    return metrics, all_predictions, all_labels, all_probs


# %% [markdown]
# ## 6. Hyperparameter Tuning with Wandb
# 
# We'll tune the following hyperparameters:
# - **Number of hidden layers**: 2, 3, 4
# - **Neurons per layer**: 64, 128, 256
# - **Learning rate**: 0.0001, 0.001, 0.01
# - **Batch size**: 16, 32, 64
# - **Dropout rate**: 0.2, 0.3, 0.4

# %%
# WandB sweep configuration (Bayesian optimization, optimize AUC)
sweep_config = {
    'method': 'bayes',
    'metric': {'name': 'test_roc_auc', 'goal': 'maximize'},
    'early_terminate': {
        'type': 'hyperband',
        'min_iter': 10
    },
    'parameters': {
        'num_layers': {'values': [2, 3, 4]},
        'neurons': {'values': [64, 128, 256]},
        'learning_rate': {'distribution': 'log_uniform', 'min': 1e-4, 'max': 1e-2},
        'batch_size': {'values': [16, 32, 64]},
        'dropout': {'values': [0.2, 0.3, 0.4]},
        'num_epochs': {'value': 100}
    }
}

print("WandB sweep configured (Bayes). Metric: test_roc_auc (maximize)")

# %%
def run_experiment(config, X_train, y_train, X_test, y_test, input_size, num_epochs=100, init_wandb=True):
    """
    Run a single experiment with given hyperparameters
    
    Parameters:
    -----------
    config : dict
        Hyperparameter configuration
    X_train, y_train : numpy arrays
        Training data
    X_test, y_test : numpy arrays
        Test data
    input_size : int
        Input feature dimension
    num_epochs : int
        Number of training epochs
    
    Returns:
    --------
    metrics : dict
        Test set performance metrics
    """
    # Initialize wandb run (or reuse existing run when called from a sweep)
    if init_wandb:
        run = wandb.init(
            project="bbb-permeability-ann",
            config=config,
            reinit=True
        )
    else:
        run = wandb.run
    
    # Split training data into train and validation sets
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
    )
    
    # Create data loaders
    train_dataset = TensorDataset(
        torch.FloatTensor(X_tr),
        torch.FloatTensor(y_tr)
    )
    val_dataset = TensorDataset(
        torch.FloatTensor(X_val),
        torch.FloatTensor(y_val)
    )
    test_dataset = TensorDataset(
        torch.FloatTensor(X_test),
        torch.FloatTensor(y_test)
    )
    
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False)
    
    # Create model
    hidden_layers = [config['neurons']] * config['num_layers']
    model = BBBPredictorANN(
        input_size=input_size,
        hidden_layers=hidden_layers,
        dropout_rate=config['dropout']
    ).to(device)
    
    # Define loss function and optimizer
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])
    
    # Train model
    print(f"\nTraining with config: {config}")
    model, history = train_model(
        model, train_loader, val_loader, criterion, optimizer, num_epochs, device
    )
    
    # Evaluate on test set
    metrics, predictions, labels, probs = evaluate_model(model, test_loader, device)
    
    # Log final metrics to wandb
    wandb.log({
        'test_accuracy': metrics['accuracy'],
        'test_precision': metrics['precision'],
        'test_recall': metrics['recall'],
        'test_f1_score': metrics['f1_score'],
        'test_roc_auc': metrics['roc_auc'],
        'test_mcc': metrics['mcc']
    })
    
    # Log confusion matrix
    cm = confusion_matrix(labels, predictions)
    wandb.log({
        "confusion_matrix": wandb.plot.confusion_matrix(
            probs=None,
            y_true=labels.astype(int),
            preds=predictions.astype(int),
            class_names=["Non-permeable", "Permeable"]
        )
    })
    
    print(f"Test Accuracy: {metrics['accuracy']:.4f}")
    print(f"Test F1-Score: {metrics['f1_score']:.4f}")
    print(f"Test ROC-AUC: {metrics['roc_auc']:.4f}")
    
    if init_wandb:
        run.finish()
    
    return metrics, model, history

print("Experiment function defined successfully!")

# %% [markdown]
# ## 7. Run Hyperparameter Tuning Experiments
# 
# Now we'll run all hyperparameter configurations and track results with Wandb.

# %%
# Run hyperparameter tuning experiments
# Note: Set to smaller number of configs for faster testing
# To run all configs, use: configs_to_run = hyperparameter_configs

# For demonstration, we'll run first 3 configs (you can change this)
configs_to_run = hyperparameter_configs[:3]  # Change to hyperparameter_configs for all

input_size = X_train_scaled.shape[1]
num_epochs = 100

results = []
best_f1 = 0
best_config = None
best_model = None

print("Starting W&B sweep (Bayesian search) for hyperparameter tuning...")

# Create sweep and start agent
sweep_id = wandb.sweep(sweep_config, project="bbb-permeability-ann")

def _sweep_run():
    # Called by wandb.agent: wandb.init is handled by agent, so we call run_experiment with init_wandb=False
    cfg = dict(wandb.config)
    metrics, model, history = run_experiment(
        config=cfg,
        X_train=X_train_scaled,
        y_train=y_train,
        X_test=X_test_scaled,
        y_test=y_test,
        input_size=input_size,
        num_epochs=cfg.get('num_epochs', 100),
        init_wandb=False
    )

    # Save run-specific results locally if desired
    results.append({**cfg, **metrics})

# Run the sweep agent (set count to number of trials you want, e.g., 20)
wandb.agent(sweep_id, function=_sweep_run, count=20)

print("W&B sweep launched. Use the W&B UI to monitor runs.")

# %% [markdown]
# ## 8. Results Analysis
# 
# Let's analyze the results from all experiments.

# %%
# Create results DataFrame
results_df = pd.DataFrame(results)

# Sort by F1-score
results_df_sorted = results_df.sort_values('f1_score', ascending=False)

print("Top 5 configurations by F1-score:")
print(results_df_sorted.head())

print(f"\n\nBest Configuration:")
print(f"{'='*60}")
for key, value in best_config.items():
    print(f"{key:20s}: {value}")
print(f"{'='*60}")
print(f"Best F1-score: {best_f1:.4f}")

# Save results to CSV
results_path = Path('../figures/BBBP/')
results_path.mkdir(parents=True, exist_ok=True)
results_df_sorted.to_csv(results_path / 'ann_hyperparameter_results.csv', index=False)
print(f"\nResults saved to {results_path / 'ann_hyperparameter_results.csv'}")


