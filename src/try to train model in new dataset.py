import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler, Subset
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from torchdiffeq import odeint
import matplotlib.pyplot as plt

# -------------------- 控制参数 --------------------
COMPARISON = 1  # 0: 只训练Neural ODE, 1: 训练并比较Neural ODE和普通NN
AGGorRAW = 0 # 0:agg, 1:raw
# -------------------- 超参数 --------------------
BATCH_SIZE    = 32
LR            = 1e-4
WEIGHT_DECAY  = 1e-2
N_EPOCHS      = 500 
PATIENCE      = 5  
ODE_HIDDEN    = 1024
NN_HIDDEN     = 1024 # 普通NN的隐藏层维度，与ODE_HIDDEN保持一致以便比较
EMBEDDING_DIM = 64
DROPOUT       = 0.1 
VAL_SIZE      = 0.2

# -------------------- 文件路径 --------------------
if AGGorRAW == 0:
    X_CSV = r"C:\Users\ryan\Desktop\ADVANCED Master\data\X_DANG_agg.csv"
    Y_CSV = r"C:\Users\ryan\Desktop\ADVANCED Master\data\y_DANG_agg.csv"
else:
    X_CSV = r"C:\Users\ryan\Desktop\ADVANCED Master\data\X_DANG_raw.csv"
    Y_CSV = r"C:\Users\ryan\Desktop\ADVANCED Master\data\y_DANG_raw.csv"
# -------------------- Dataset定义 --------------------
class PrepDataset(Dataset):
    def __init__(self, X_csv, y_csv):
        X_df = pd.read_csv(X_csv, index_col=0)
        y_df = pd.read_csv(y_csv, index_col=0)

        self.conditions = X_df.idxmax(axis=1).values
        self.condition_to_idx = {cond: i for i, cond in enumerate(sorted(set(self.conditions)))}
        self.condition_idx = torch.tensor([self.condition_to_idx[c] for c in self.conditions], dtype=torch.long)

        self.y = torch.from_numpy(y_df.values).float()
        if self.y.ndim == 1:
            self.y = self.y.unsqueeze(1) # 确保y是 [N, num_outputs]

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.condition_idx[idx], self.y[idx], self.conditions[idx]

# -------------------- Neural ODE模型定义 --------------------
class ODEFunc(nn.Module):
    def __init__(self, feature_dim, hidden_dim, dropout=DROPOUT):
        super().__init__()
        self.linear_in = nn.Linear(feature_dim, hidden_dim)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.linear_hidden = nn.Linear(hidden_dim, hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.linear_out = nn.Linear(hidden_dim, feature_dim)
        self.activation = nn.GELU()
        self.dropout_layer = nn.Dropout(dropout) 

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, t, x):
        residual = x
        out = self.activation(self.linear_in(x))
        out = self.dropout_layer(self.norm1(out))
        out = self.activation(self.linear_hidden(out))
        out = self.dropout_layer(self.norm2(out))
        out = self.linear_out(out)
        return out * 0.5 + residual * 0.5

class NeuralODE(nn.Module):
    def __init__(self, num_conditions, embedding_dim, ode_hidden_dim, output_dim, solver='dopri5', dropout=DROPOUT):
        super().__init__()
        self.embedding = nn.Embedding(num_conditions, embedding_dim)
        self.input_layer = nn.Sequential(
            nn.Linear(embedding_dim, ode_hidden_dim),
            nn.LayerNorm(ode_hidden_dim),
            nn.GELU()
        )
        self.odefunc = ODEFunc(ode_hidden_dim, hidden_dim=ode_hidden_dim, dropout=dropout)
        self.output_layer = nn.Sequential(
            nn.LayerNorm(ode_hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(ode_hidden_dim, output_dim)
        )
        self.integration_time = torch.tensor([0.0, 1.0])
        self.solver = solver

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Embedding):
                nn.init.xavier_normal_(m.weight)


    def forward(self, x_idx):
        x = self.embedding(x_idx)
        x = self.input_layer(x)
        integration_time = self.integration_time.to(x.device)
        # Explicitly use options for rk4 if needed, or handle other solvers
        if self.solver in ['rk4', 'dopri5', 'adams']:
             out = odeint(self.odefunc, x, integration_time, method=self.solver, options=dict(step_size=0.1) if self.solver=='rk4' else None)[-1]
        else: # For adaptive solvers that might not take options or specific ones
             out = odeint(self.odefunc, x, integration_time, method=self.solver)[-1]

        out = self.output_layer(out)
        return out

# -------------------- 普通神经网络模型定义 --------------------
class NormalNN(nn.Module):
    def __init__(self, num_conditions, embedding_dim, nn_hidden_dim, output_dim, dropout=DROPOUT):
        super().__init__()
        self.embedding = nn.Embedding(num_conditions, embedding_dim)
        self.layers = nn.Sequential(
            nn.Linear(embedding_dim, nn_hidden_dim),
            nn.LayerNorm(nn_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(nn_hidden_dim, nn_hidden_dim),
            nn.LayerNorm(nn_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(nn_hidden_dim, output_dim)
        )
        # Xavier initialization
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Embedding):
                nn.init.xavier_normal_(m.weight)


    def forward(self, x_idx):
        x = self.embedding(x_idx)
        x = self.layers(x)
        return x

# -------------------- 准备数据 --------------------
try:
    ds = PrepDataset(X_CSV, Y_CSV)
except FileNotFoundError:
    print(f"错误: 找不到数据文件。请检查路径:\nX_CSV: {X_CSV}\nY_CSV: {Y_CSV}")
    exit()

all_idx = np.arange(len(ds))
train_idx, val_idx = train_test_split(all_idx, test_size=VAL_SIZE, random_state=42, shuffle=True)

train_conditions_list = [ds.conditions[i] for i in train_idx] # Renamed to avoid conflict
unique_train_conditions, counts_train_conditions = np.unique(train_conditions_list, return_counts=True)
freq_train_conditions = dict(zip(unique_train_conditions, counts_train_conditions))


train_weights = [1.0 / freq_train_conditions[c] for c in train_conditions_list if c in freq_train_conditions]

valid_train_indices_for_sampler = [i for i, c in zip(train_idx, train_conditions_list) if c in freq_train_conditions]
if len(valid_train_indices_for_sampler) != len(train_idx):
    print("警告: 某些训练样本的条件在频率计算中缺失，已调整采样器。")
    

train_sampler = WeightedRandomSampler(train_weights, num_samples=len(train_weights), replacement=True)

train_loader = DataLoader(Subset(ds, train_idx), batch_size=BATCH_SIZE, sampler=train_sampler)
val_loader   = DataLoader(Subset(ds, val_idx), batch_size=BATCH_SIZE, shuffle=False)

# -------------------- 模型 + 训练准备 --------------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

num_unique_conditions = len(set(ds.conditions)) # Use all unique conditions for embedding size
output_dim = ds.y.shape[1]

# --- Neural ODE Model ---
model_ode = NeuralODE(num_conditions=num_unique_conditions,
                      embedding_dim=EMBEDDING_DIM,
                      ode_hidden_dim=ODE_HIDDEN,
                      output_dim=output_dim,
                      dropout=DROPOUT).to(device)
optimizer_ode = optim.Adam(model_ode.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
criterion = nn.MSELoss() # Shared criterion

train_losses_ode, val_losses_ode = [], []
best_val_loss_ode = float('inf')
patience_cnt_ode = 0
early_stop_ode = False
model_ode_saved_path = 'best_model_ode.pth'


# --- Normal NN Model (if comparison is enabled) ---
if COMPARISON == 1:
    model_nn = NormalNN(num_conditions=num_unique_conditions,
                        embedding_dim=EMBEDDING_DIM,
                        nn_hidden_dim=NN_HIDDEN,
                        output_dim=output_dim,
                        dropout=DROPOUT).to(device)
    optimizer_nn = optim.Adam(model_nn.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    train_losses_nn, val_losses_nn = [], []
    best_val_loss_nn = float('inf')
    patience_cnt_nn = 0
    early_stop_nn = False
    model_nn_saved_path = 'best_model_nn.pth'

# -------------------- 训练循环 --------------------
actual_epochs = 0
for epoch in range(1, N_EPOCHS + 1):
    actual_epochs = epoch
    epoch_train_loss_ode, epoch_val_loss_ode = 0.0, 0.0
    epoch_train_loss_nn, epoch_val_loss_nn = 0.0, 0.0

    # --- Train Neural ODE ---
    if not early_stop_ode:
        model_ode.train()
        current_train_loss_ode = 0.0
        for condition_idx, yb, _ in train_loader:
            condition_idx, yb = condition_idx.to(device), yb.to(device)
            optimizer_ode.zero_grad()
            preds_ode = model_ode(condition_idx)
            loss_ode = criterion(preds_ode, yb)
            loss_ode.backward()
            optimizer_ode.step()
            current_train_loss_ode += loss_ode.item() * condition_idx.size(0)
        epoch_train_loss_ode = current_train_loss_ode / len(train_idx)
        train_losses_ode.append(epoch_train_loss_ode)

    # --- Train Normal NN (if comparison) ---
    if COMPARISON == 1 and not early_stop_nn:
        model_nn.train()
        current_train_loss_nn = 0.0
        for condition_idx, yb, _ in train_loader: # Re-iterate or store data? Re-iterate for simplicity
            condition_idx, yb = condition_idx.to(device), yb.to(device)
            optimizer_nn.zero_grad()
            preds_nn = model_nn(condition_idx)
            loss_nn = criterion(preds_nn, yb)
            loss_nn.backward()
            optimizer_nn.step()
            current_train_loss_nn += loss_nn.item() * condition_idx.size(0)
        epoch_train_loss_nn = current_train_loss_nn / len(train_idx)
        train_losses_nn.append(epoch_train_loss_nn)

    # --- Validate Neural ODE ---
    if not early_stop_ode:
        model_ode.eval()
        current_val_loss_ode = 0.0
        with torch.no_grad():
            for condition_idx, yb, _ in val_loader:
                condition_idx, yb = condition_idx.to(device), yb.to(device)
                preds_ode = model_ode(condition_idx)
                current_val_loss_ode += criterion(preds_ode, yb).item() * condition_idx.size(0)
        epoch_val_loss_ode = current_val_loss_ode / len(val_idx)
        val_losses_ode.append(epoch_val_loss_ode)

        if epoch_val_loss_ode < best_val_loss_ode:
            best_val_loss_ode = epoch_val_loss_ode
            torch.save(model_ode.state_dict(), model_ode_saved_path)
            patience_cnt_ode = 0
        else:
            patience_cnt_ode += 1
            if patience_cnt_ode >= PATIENCE:
                print(f"Neural ODE early stopping at epoch {epoch}")
                early_stop_ode = True
    elif early_stop_ode and len(val_losses_ode) < epoch: # Fill in losses if already stopped
         val_losses_ode.append(val_losses_ode[-1] if val_losses_ode else float('inf'))
         if len(train_losses_ode) < epoch: train_losses_ode.append(train_losses_ode[-1] if train_losses_ode else float('inf'))


    # --- Validate Normal NN (if comparison) ---
    if COMPARISON == 1 and not early_stop_nn:
        model_nn.eval()
        current_val_loss_nn = 0.0
        with torch.no_grad():
            for condition_idx, yb, _ in val_loader: # Re-iterate
                condition_idx, yb = condition_idx.to(device), yb.to(device)
                preds_nn = model_nn(condition_idx)
                current_val_loss_nn += criterion(preds_nn, yb).item() * condition_idx.size(0)
        epoch_val_loss_nn = current_val_loss_nn / len(val_idx)
        val_losses_nn.append(epoch_val_loss_nn)

        if epoch_val_loss_nn < best_val_loss_nn:
            best_val_loss_nn = epoch_val_loss_nn
            torch.save(model_nn.state_dict(), model_nn_saved_path)
            patience_cnt_nn = 0
        else:
            patience_cnt_nn += 1
            if patience_cnt_nn >= PATIENCE:
                print(f"Normal NN early stopping at epoch {epoch}")
                early_stop_nn = True
    elif COMPARISON == 1 and early_stop_nn and len(val_losses_nn) < epoch: # Fill in losses if already stopped
        val_losses_nn.append(val_losses_nn[-1] if val_losses_nn else float('inf'))
        if len(train_losses_nn) < epoch: train_losses_nn.append(train_losses_nn[-1] if train_losses_nn else float('inf'))


    # --- Print Epoch Summary ---
    print_msg = f"Epoch {epoch:03d} | "
    if not early_stop_ode or epoch_train_loss_ode != 0: # Print if not stopped or if has valid loss for this epoch
         print_msg += f"ODE Train MSE: {epoch_train_loss_ode:.4f} | ODE Val MSE: {epoch_val_loss_ode:.4f} | "
    if COMPARISON == 1 and (not early_stop_nn or epoch_train_loss_nn != 0):
         print_msg += f"NN Train MSE: {epoch_train_loss_nn:.4f} | NN Val MSE: {epoch_val_loss_nn:.4f}"
    print(print_msg.strip().strip("|").strip())


    # --- Check if all models early stopped ---
    if COMPARISON == 1:
        if early_stop_ode and early_stop_nn:
            print("Both models early stopped.")
            break
    else: # Only ODE
        if early_stop_ode:
            break
# Ensure loss lists are of the same length as actual_epochs for plotting
while len(train_losses_ode) < actual_epochs: train_losses_ode.append(train_losses_ode[-1] if train_losses_ode else float('inf'))
while len(val_losses_ode) < actual_epochs: val_losses_ode.append(val_losses_ode[-1] if val_losses_ode else float('inf'))
if COMPARISON == 1:
    while len(train_losses_nn) < actual_epochs: train_losses_nn.append(train_losses_nn[-1] if train_losses_nn else float('inf'))
    while len(val_losses_nn) < actual_epochs: val_losses_nn.append(val_losses_nn[-1] if val_losses_nn else float('inf'))


# -------------------- 绘制损失曲线 --------------------
plt.figure(figsize=(12, 7))
plt.plot(range(1, actual_epochs + 1), train_losses_ode[:actual_epochs], label='ODE Train MSE', color='blue', linestyle='--')
plt.plot(range(1, actual_epochs + 1), val_losses_ode[:actual_epochs], label='ODE Val MSE', color='blue')

if COMPARISON == 1:
    plt.plot(range(1, actual_epochs + 1), train_losses_nn[:actual_epochs], label='NN Train MSE', color='red', linestyle='--')
    plt.plot(range(1, actual_epochs + 1), val_losses_nn[:actual_epochs], label='NN Val MSE', color='red')

plt.xlabel('Epoch')
plt.ylabel('MSE Loss')
plt.title('Training and Validation Loss Comparison' if COMPARISON == 1 else 'Training and Validation Loss (Neural ODE)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

print("训练完成。")
if not early_stop_ode or best_val_loss_ode != float('inf'):
    print(f"最佳Neural ODE模型 ({model_ode_saved_path}) 验证MSE: {best_val_loss_ode:.4f}")
if COMPARISON == 1 and (not early_stop_nn or best_val_loss_nn != float('inf')):
    print(f"最佳Normal NN模型 ({model_nn_saved_path}) 验证MSE: {best_val_loss_nn:.4f}")