import torch
import torch.nn as nn

class DQNMetricsMLP(nn.Module):
    def __init__(self, input_dim=256, hidden_dim=512, output_dim=4):
        super(DQNMetricsMLP, self).__init__()  # ← add this line
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc4 = nn.Linear(hidden_dim // 2, output_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
    
    def forward(self, x):
        x = torch.relu(self.bn1(self.fc1(x)))
        x = torch.relu(self.bn2(self.fc2(x)))
        x = torch.relu(self.fc3(x))
        return self.fc4(x)

class DQNCNNAgent(nn.Module):
    def __init__(self, output_dim=4):
        super(DQNCNNAgent, self).__init__()
        self.conv1 = nn.Conv2d(16, 64, kernel_size=2, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=2, padding=1)
        self.conv3 = nn.Conv2d(128, 128, kernel_size=2)
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(128)
        self.fc1 = nn.Linear(3200, 256)  # FIX: was 128*4*4=2048, correct is 3200
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, output_dim)


    def forward(self, x):
        x = x.view(-1, 16, 4, 4)
        x = torch.relu(self.bn1(self.conv1(x)))
        x = torch.relu(self.bn2(self.conv2(x)))
        x = torch.relu(self.conv3(x))
        x = x.view(x.size(0), -1)  # flatten → should be 3200
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)