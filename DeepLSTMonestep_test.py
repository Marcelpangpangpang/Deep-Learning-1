import torch
import matplotlib.pyplot as plt
import torch.optim as optim
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error
from scipy.io import loadmat
from sklearn.preprocessing import MinMaxScaler
import torch.nn as nn

class DeepLSTMOneStep(nn.Module): #code waarop model is gertraind
    def __init__(self, input_size=1, hidden_size=128, num_layers=2, dropout=0.0):
        super(DeepLSTMOneStep, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0  
        )

        self.head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 1)
        )

    def forward(self, x):
        lstm_out, _ = self.lstm(x)        
        out = lstm_out[:, -1, :]           
        out = self.head(out)              
        return out   
    
#model creeren, inlezen en op testmodus zetten
model = DeepLSTMOneStep(input_size=1, hidden_size=248, num_layers=3, dropout=0.0)
model.load_state_dict(torch.load('/Users/mariskacordus/Downloads/DeepLSTMonestep.pth'))
model.eval()

data = loadmat('/Users/mariskacordus/Desktop/DL/Xtrain.mat')
data = data['Xtrain'].squeeze()
data = data.reshape(-1, 1)

#testdata code toevoegem
testdata = loadmat('/Users/mariskacordus/Desktop/DL/Xtest.mat')
testdata = testdata['Xtest'].squeeze()
testdata = testdata.reshape(-1, 1)

#code waarop model is gertraind
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(data)

prediction_range = 200   
predictions_scaled = []

input_len=200

last_sequence = torch.tensor(data_scaled[-input_len:], dtype=torch.float32).unsqueeze(0)

with torch.no_grad():
    for _ in range(prediction_range):
        pred = model(last_sequence)
        predictions_scaled.append(pred.item())
        new_val = pred.unsqueeze(2)  # shape: [1, 1, 1]
        last_sequence = torch.cat((last_sequence[:, 1:, :], new_val), dim=1)

predictions_np = np.array(predictions_scaled).reshape(-1, 1)
predictions_unscaled = scaler.inverse_transform(predictions_np).squeeze()

#MAE en MSE berekenen
testdata_scaled = scaler.fit_transform(testdata)
mae = mean_absolute_error(testdata_scaled, predictions_np)
mse = mean_squared_error(testdata_scaled, predictions_np)
print("MAE van DEEP is ")
print(mae)
print("MSE is DEEP is ")
print(mse)

# Plotten 
plt.figure(figsize=(10, 5))
plt.scatter(np.arange(len(data), len(data) + len(predictions_unscaled)), predictions_unscaled, 
                label='Predicted values (200 steps)', color='red', s=2.5)
plt.scatter(np.arange(len(data), len(data) + len(testdata)), testdata, 
                label='Test values (200 steps)', color='green', s=2.5)
plt.title("Recursive Forecast from Deep LSTM Model")
plt.xlabel("Time Step")
plt.ylabel("Value")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
