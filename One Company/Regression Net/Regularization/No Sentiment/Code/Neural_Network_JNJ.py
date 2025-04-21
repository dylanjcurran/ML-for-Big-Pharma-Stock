import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import seaborn as sns


#--- Variables ---

CSV_FILE = "JNJ.csv"

#--- Neural Net Class ---
class Net(nn.Module):
    def __init__(self, input_dim):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_dim, 8)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(8,4)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(4, 2)
        self.relu3 = nn.ReLU()
        self.output = nn.Linear(2, 1)
        #self.sigmoid = nn.Sigmoid()  

    def forward(self, x):
        x = self.relu1(self.fc1(x))
        x = self.relu2(self.fc2(x))
        x = self.relu3(self.fc3(x))
        x = self.output(x)
        return x


#--- Main Script ---

#Load CSV
df = pd.read_csv("JNJ.csv")
df.columns = df.columns.str.strip()

drop_features = ['Simple Moving Average', 'Bollinger Upper Band', 'Bollinger Lower Band']
features = [col for col in df.columns if col not in drop_features and col not in ["Output", "StartDate", "EndDate", "Ticker", "Company"]]


#Shift to Prevent Leakage
df[features] = df[features].shift(1)

#Drop NA values
df.dropna(inplace=True)

#Sort Chronologically
df = df.sort_values(by=["StartDate", "Ticker"]).reset_index(drop=True)

# Drop metadata columns
df = df.drop(columns=["Ticker", "Company", "StartDate", "EndDate", "Simple Moving Average", "Bollinger Upper Band", "Bollinger Lower Band"])

# Split features and target
X = df.drop(columns=["Output"])
y = df["Output"]

# 80% train, 20% test
train_size = int(len(df) * 0.8)
train_df = df[:train_size]
test_df = df[train_size:]

X_train = train_df[features]
y_train = train_df["Output"]

X_test = test_df[features]
y_test = test_df["Output"]

#Scale Values
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

#Convert to PyTorch Tensors
X_train_tensor = torch.tensor(np.array(X_train_scaled), dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)

X_test_tensor = torch.tensor(np.array(X_test_scaled), dtype=torch.float32)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).view(-1, 1)

#Making Instance of Net
model = Net(X_train_tensor.shape[1])

#Loss Function
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

#Training
epochs = 1000
train_losses = []
test_losses = []

for epoch in range(epochs):
    model.train()
    train_preds = model(X_train_tensor)
    train_loss = criterion(train_preds, y_train_tensor)

    optimizer.zero_grad()
    train_loss.backward()
    optimizer.step()

    # Evaluate test loss (no gradients)
    model.eval()
    with torch.no_grad():
        test_preds = model(X_test_tensor)
        test_loss = criterion(test_preds, y_test_tensor)

    # Store losses
    train_losses.append(train_loss.item())
    test_losses.append(test_loss.item())

    if epoch % 10 == 0:
        print(f"Epoch {epoch} | Train Loss: {train_loss.item():.4f} | Test Loss: {test_loss.item():.4f}")
        
#Visualization
plt.figure(figsize=(8, 5))
plt.plot(train_losses, label='Training Loss', color='blue')
plt.plot(test_losses, label='Test Loss', color='red')
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.title("Training vs Test Loss")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("Loss_Over_Epochs.png", dpi=300, bbox_inches='tight')
plt.close()

#Model Evaluation
model.eval()
with torch.no_grad():
    preds = model(X_test_tensor).numpy().flatten()
    actuals = y_test_tensor.numpy().flatten()

#Evaluation Plot
plt.figure(figsize=(6, 6))
plt.scatter(actuals, preds, alpha=0.6)
plt.plot([actuals.min(), actuals.max()], [actuals.min(), actuals.max()], 'r--')  # Ideal line
plt.xlabel("Actual Output")
plt.ylabel("Predicted Output")
plt.title("Actual vs. Predicted")
plt.grid(True)
plt.tight_layout()
plt.savefig("Evaluation.png", dpi=300, bbox_inches='tight')
plt.close()

#Residual Plot
residuals = preds - actuals

plt.figure(figsize=(8, 5))
plt.scatter(actuals, residuals, alpha=0.5)
plt.axhline(0, color='red', linestyle='--')
plt.xlabel("Actual Output")
plt.ylabel("Residual (Prediction - Actual)")
plt.title("Residual Plot")
plt.grid(True)
plt.tight_layout()
plt.savefig("Residuals.png", dpi=300, bbox_inches='tight')
plt.close()

#Checking Feature Correlations
correlation_df = X.copy()
correlation_df["Output"] = y
correlation_df = correlation_df.dropna()
correlation_matrix = correlation_df.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Feature Correlation Heatmap (Shifted Inputs)")
plt.tight_layout()
plt.savefig("Correlation_Heatmap.png", dpi=300)
plt.close()


#Metrics
y_pred = model(X_test_tensor).detach().numpy().flatten()
y_true = y_test_tensor.numpy().flatten()

mse = mean_squared_error(y_true, y_pred)
mae = mean_absolute_error(y_true, y_pred)
r2 = r2_score(y_true, y_pred)

print(f"MSE: {mse:.4f}")
print(f"MAE: {mae:.4f}")
print(f"R²: {r2:.4f}")

naive_pred = [y_train.mean()] * len(y_test)

print("Naive MSE:", mean_squared_error(y_test, naive_pred))
print("Naive R²:", r2_score(y_test, naive_pred))


#Saving Model
"""torch.save(model.state_dict(), "neural_net_model.pth")"""
