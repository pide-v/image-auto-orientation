import torch
import torch.nn as nn
import torchvision.models as models
import torch.optim as optim

from torchvision import datasets, transforms
from torch.utils.data import DataLoader

model = models.resnet50(pretrained=True)

#replacing resnet's fc layer (1000 classes) with 4 classes.
model.fc = nn.Linear(model.fc.in_features, 4)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


#freezing all the weights except for fc layer.
for param in model.parameters():
	param.requires_grad = False

for param in model.fc.parameters():
	param.requires_grad = True


transform = transforms.Compose([
    transforms.Resize((160, 160)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

train_dataset = datasets.ImageFolder('/home/pide/aml/image-auto-orientation/ss-dataset/train', transform=transform)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
#training loop
num_epochs = 15
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f'Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(train_loader):.4f}')

print('Training complete!')


# Save model
torch.save(model.state_dict(), 'resnet_orientation.pth')

# Load model
model.load_state_dict(torch.load('resnet_orientation.pth'))
model.eval()  # Set to evaluation mode

test_dataset = datasets.ImageFolder('/home/pide/aml/image-auto-orientation/ss-dataset/test', transform=transform)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

correct = 0
total = 0
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

accuracy = 100 * correct / total
print(f"Test Accuracy: {accuracy:.2f}%")