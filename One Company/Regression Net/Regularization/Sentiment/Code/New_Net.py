import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np

# Load data
df = pd.read_csv("JNJSentReg.csv")
features = [col for col in df.columns if col not in ["Output", "StartDate", "EndDate", "Company", "Ticker"]]
X = df[features].values
y = df["Output"].values.reshape(-1, 1)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalize
scaler_X = StandardScaler()
scaler_y = StandardScaler()
X_train_scaled = scaler_X.fit_transform(X_train)
X_test_scaled = scaler_X.transform(X_test)
y_train_scaled = scaler_y.fit_transform(y_train)
y_test_scaled = scaler_y.transform(y_test)

# Convert to tensors
X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train_scaled, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test_scaled, dtype=torch.float32)
def evaluate_model(hidden1, hidden2, dropout_rate, weight_decay, lr=1e-3, epochs=300):
    input_dim = X_train_tensor.shape[1]

    class SimpleNet(nn.Module):
        def __init__(self):
            super(SimpleNet, self).__init__()
            self.fc1 = nn.Linear(input_dim, hidden1)
            self.relu1 = nn.ReLU()
            self.drop1 = nn.Dropout(dropout_rate)
            self.fc2 = nn.Linear(hidden1, hidden2)
            self.relu2 = nn.ReLU()
            self.drop2 = nn.Dropout(dropout_rate)
            self.out = nn.Linear(hidden2, 1)

        def forward(self, x):
            x = self.drop1(self.relu1(self.fc1(x)))
            x = self.drop2(self.relu2(self.fc2(x)))
            return self.out(x)

    model = SimpleNet()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train_tensor)
        loss = criterion(outputs, y_train_tensor)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Clip gradients
        optimizer.step()

    model.eval()
    with torch.no_grad():
        preds = model(X_test_tensor).numpy()
        true = y_test_tensor.numpy()

        if np.isnan(preds).any() or np.isnan(true).any():
            print(f"⚠️ NaNs detected in predictions for config (h1={hidden1}, h2={hidden2}, dropout={dropout_rate}, wd={weight_decay}) — skipping.")
            return {
                "Hidden1": hidden1,
                "Hidden2": hidden2,
                "Dropout": dropout_rate,
                "WeightDecay": weight_decay,
                "MSE": np.nan,
                "MAE": np.nan,
                "R²": np.nan
            }

        preds_unscaled = scaler_y.inverse_transform(preds)
        true_unscaled = scaler_y.inverse_transform(true)

        mse = mean_squared_error(true_unscaled, preds_unscaled)
        mae = mean_absolute_error(true_unscaled, preds_unscaled)
        r2 = r2_score(true_unscaled, preds_unscaled)

    return {
        "Hidden1": hidden1,
        "Hidden2": hidden2,
        "Dropout": dropout_rate,
        "WeightDecay": weight_decay,
        "MSE": mse,
        "MAE": mae,
        "R²": r2
    }

# Configurations to try
configs = [
    (4, 2, 0.3, 1e-4),
    (8, 4, 0.3, 1e-4),
    (16, 8, 0.2, 1e-4),
    (8, 4, 0.5, 1e-3),
    (16, 8, 0.3, 1e-3)
]

results = [evaluate_model(*cfg) for cfg in configs]
results_df = pd.DataFrame(results)
print(results_df)
