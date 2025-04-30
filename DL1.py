import scipy.io
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

mat = scipy.io.loadmat('/Users/mariskacordus/Desktop/DL/Xtrain.mat')
#print(mat)
Xtrain255 = mat['Xtrain']
Xtrain255 = Xtrain255.flatten() #makkelijker leesbaar, 1D
Xtrain = Xtrain255/255 #normaliseren; 255 omdat uint8 max is 255
#print(Xtrain_flat)
def create_sequences(data, seq_length):
    xs = []
    ys = []
    for i in range(len(data) - seq_length):
        x = data[i:i+seq_length]  # seq_length punten input
        y = data[i+seq_length]    # 1 punt na het aantal punten seq_length output (het volgende punt)
        xs.append(x)
        ys.append(y)
    X_seq = torch.from_numpy(np.array(xs)).float().unsqueeze(-1) #omzetten naar PyTorch tensors
    y_seq = torch.from_numpy(np.array(ys)).float().unsqueeze(-1)           

    return X_seq, y_seq

#seq_length = 10 #how many past data point can be used, to be tuned!
#X_d1, Y_d1=create_sequences(Xtrain, seq_length)
#print(X_d1[0], Y_d1[0])

class LSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=50, output_size=1):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        out = lstm_out[:, -1, :]  # alleen laatste tijdstap gebruiken
        out = self.fc(out)  # naar 1 output value
        return out

def training(model, X_seq, Y_seq):
    criterion = nn.MSELoss()  # Mean Squared Error als verliesfunctie
    optimizer = optim.Adam(model.parameters(), lr=0.01)  # Adam optimizer

    num_epochs = 25

    for epoch in range(num_epochs):
        model.train()
        output = model(X_seq)  # voorspel op alle inputs
        loss = criterion(output, Y_seq)  # vergelijk met echte antwoorden

        optimizer.zero_grad()
        loss.backward()  # backpropagation
        optimizer.step()  # werk weights bij

        final_loss = loss.item()
        if (epoch+1) % 10 == 0: #kijken wat het model leert
            print(f'Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}')

    return final_loss

best_loss = float('inf')
best_seq_length = None

#for seq_length in range(85, 95, 1):  
    # probeer seq_length 
    # 5-100 met 5 gaf 95;
    # 95-205 met 10 gaf 95;
    # 80-120 met 2 gaf 94;
    # 92-96 met 1 gaf 92;
    # 85-95 met 1 gaf 90, 94, 89

#    X_seq, y_seq = create_sequences(Xtrain, seq_length)
#    model = LSTMModel()
#    final_loss = training(model, X_seq, y_seq) 
#
#    if final_loss < best_loss:
#        best_loss = final_loss
#        best_seq_length = seq_length

#print(f"\nBeste seq_length: {best_seq_length} met loss {best_loss:.4f}")

seq_length = 90
X_seq, y_seq = create_sequences(Xtrain, seq_length)
model = LSTMModel()

model.eval()


### Hier bereken je de i volgende waardes. Eerste waarde lijkt goed te gaan, maar die daarna niet 
i=1
while (i < 5):
    last_seq = torch.from_numpy(Xtrain[-seq_length:]).float().unsqueeze(0).unsqueeze(-1)   
    with torch.no_grad():
        pred = model(last_seq)
        pred = torch.clamp(pred, 0.0, 1.0) 
        print(pred)
    Xtrain = np.append(Xtrain, [pred.item()])
    i=i+1
new_Xtrain255 = Xtrain*255
print(new_Xtrain255[-5:], len(new_Xtrain255))

