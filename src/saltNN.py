import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np


# need to find a place to train
# First attempt
# def needs a rewrite...  
torch.manual_seed(42)
np.random_seed(42)

inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
outputs = np.array([[0], [1], [1], [0]])

inputs_tensor = torch.from_numpy(inputs).float()
outputs_tensor = torch.from_numpy(outputs).float()

class SaltBot(nn.Module):
    def __init__(self):
        super(SaltBot, self).__init__()
        self.fc1 = nn.Linear(2, 2)
        self.fc2 = nn.Linear(2, 1)

    def forward(self, x):
        x = torch.sigmoid(self.fc1(x))
        x = self.fc2(x)
        return x

model = SaltBot()
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)

for epoch in range(1000):
    optimizer.zero_grad()
    outputs_pred = model(inputs_tensor)
    loss = criterion(outputs_pred, outputs_tensor)
    loss.backward()
    optimizer.step()
    if epoch % 100 == 0:
        print(f'Epoch {epoch+1}, Loss: {loss.item():.4f}')

with torch.no_grad():
    outputs_pred = model(inputs_tensor)
    print('SaltBot Predictions:')
    print(outputs_pred)

#adding a function to see if the model is running to interact with the interface
def saltBotRunning ():
    with outputs_pred == True:
        return True