# federated_attack_comparison.py
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from copy import deepcopy
import numpy as np

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyper-parameters
num_clients = 5
num_rounds = 10
frac_malicious = 0.4
local_epochs = 2
batch_size = 64
learning_rate = 0.01

# Define transformations
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Load CIFAR-10
train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                             download=True, transform=transform)
test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                            download=True, transform=transform)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1000, shuffle=False)

# Define the model
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 8 * 8, 512)
        self.fc2 = nn.Linear(512, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 8 * 8)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Attack functions
def apply_attack(data, target, attack_type):
    if attack_type == "targeted":
        target = torch.where(target == 0, torch.tensor(1), target)
    elif attack_type == "random_label":
        target = torch.randint(0, 10, target.shape)
    elif attack_type == "random_input":
        data = data + 0.2 * torch.randn_like(data)
    return data, target

def train_local(model, optimizer, criterion, data_loader, attack_type, is_malicious):
    model.train()
    for epoch in range(local_epochs):
        for data, target in data_loader:
            data, target = data.to(device), target.to(device)
            if is_malicious:
                data, target = apply_attack(data, target, attack_type)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            if attack_type == "inverted" and is_malicious:
                loss = 1.0 / (loss + 1e-6)
            loss.backward()
            optimizer.step()
    return model

def evaluate(model):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)
    return correct / total

# Run federated training for each attack type
attack_types = ["none", "targeted", "random_label", "random_input", "inverted"]
results = {}

for attack_type in attack_types:
    print(f"\n=== Attack: {attack_type} ===")
    global_model = CNN().to(device)
    global_model.load_state_dict(CNN().state_dict())
    acc_list = []

    for rnd in range(num_rounds):
        local_models = []
        for i in range(num_clients):
            local_model = deepcopy(global_model)
            optimizer = optim.SGD(local_model.parameters(), lr=learning_rate)
            criterion = nn.CrossEntropyLoss()
            is_malicious = (i < int(num_clients * frac_malicious)) and attack_type != "none"
            local_models.append(train_local(local_model, optimizer, criterion, train_loader, attack_type, is_malicious))

        # Federated averaging
        global_dict = global_model.state_dict()
        for k in global_dict.keys():
            global_dict[k] = torch.stack([local_models[i].state_dict()[k].float() for i in range(num_clients)], 0).mean(0)
        global_model.load_state_dict(global_dict)

        acc = evaluate(global_model)
        acc_list.append(acc)
        print(f"Round {rnd+1}/{num_rounds}, Accuracy: {acc:.4f}")

    results[attack_type] = acc_list

# Plot all accuracies
plt.figure(figsize=(10, 6))
for attack_type, acc_list in results.items():
    plt.plot(range(1, num_rounds + 1), acc_list, marker='o', label=attack_type)

plt.xlabel("Communication Round")
plt.ylabel("Accuracy")
plt.title("CIFAR-10: Comparaison des attaques")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("/mnt/data/courbes_comparatives.png")
plt.show()
