import torch
import os
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
import numpy as np
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
import torchvision.models as models #importing the resnet model from torchvision

#Resnet Model
transform = transforms.Compose([
    transforms.Resize((224, 224)),#ResNet-18 expects 224x224 input size
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.3444, 0.3803, 0.4078],
        std=[0.1995, 0.1348, 0.1138]
    )
])

#Load the dataset
dataset = ImageFolder(root=r'C:\Users\john_mg\U-net_segmentation\EuroSAT\2750',
                      transform = transform)

#Split the dataset into training, validation and test sets
total = len(dataset)
train_size = int(0.7 * total)
val_size = int(0.15 * total)
test_size = total - train_size - val_size
train_set, val_set, test_set = random_split(dataset, [train_size, val_size, test_size])

#Create DataLoaders for each sets
train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
val_loader = DataLoader(val_set, batch_size=32, shuffle=False)
test_loader = DataLoader(test_set, batch_size=32, shuffle=False)

print(f"Total images: {total}")
print(f"Training images: {len(train_set)}")
print(f"Validation images: {len(val_set)}")
print(f"Testing images: {len(test_set)}")

#Check one batch
images, labels = next(iter(train_loader))
print(f"\nBatch image shape:{images.shape}")
print(f"\nBatch label shape:{labels.shape}")

#Build the ResNet model
class ResNetClassifier(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()

        #Load the pre-trained ResNet-18 model
        self.model = models.resnet18(weights='IMAGENET1K_V1')

        #Replace the final fully connected layer to match our number of classes
        for param in self.model.parameters():
            param.requires_grad = False #Freeze the pre-trained layers
        #unfreeze the last two blocks of the ResNet model to allow fine-tuning
        for param in self.model.layer3.parameters():
            param.requires_grad = True
        for param in self.model.layer4.parameters():
            param.requires_grad = True
            
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

    def forward(self, x):
        return self.model(x) #Forward pass through the ResNet model
    
#Instantiate the model
model = ResNetClassifier()

#Verify the final layer
print(f"\nFinal layer: {model.model.fc}")

#test with one batch
with torch.no_grad():
    output = model(images)
    print(f"Output shape: {output.shape}")

#Count parameters
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"\nTotal parameters: {total_params}")
print(f"Trainable parameters: {trainable_params}")

#Train the model
criterion = nn.CrossEntropyLoss()
#The more pretrained weights you're updating, the lower the learning rate should be.
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)#lr=1e-4 is standard learning rate for fine-tuning pretrained models.

train_losses = []
val_losses = []
val_accuracies = []

EPOCHS = 15
'''
Stores the lowest validation loss seen so far
Starts at infinity so the first epoch always counts as an improvement
Gets updated every time a better model is found
'''
best_val_loss = float('inf') # Initialize best validation loss to infinity for early stopping
patience = 3 # Number of epochs to wait for an improvement before stopping
'''
Counter to track how many epochs have passed without improvement
Resets to 0 when a better model is found
When it reaches patience → trigger early stopping
'''
counter = 0 

TRAIN = True
if TRAIN:
    for epoch in range(EPOCHS):
        model.train()
        total_train_loss = 0
        for images, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()#backpropagation
            optimizer.step()#update weights
            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        model.eval()
        total_val_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                outputs = model(images)
                loss = criterion(outputs, labels)
                total_val_loss += loss.item()

                predicted = outputs.argmax(dim=1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)
        
        avg_val_loss = total_val_loss / len(val_loader)
        val_accuracy = correct / total * 100
        val_losses.append(avg_val_loss)
        val_accuracies.append(val_accuracy)
        print(f"Epoch {epoch+1}/{EPOCHS} - Train Loss: {avg_train_loss:.4f} - Val Loss: {avg_val_loss:.4f} - Val Accuracy: {val_accuracy:.2f}%")
        # Early Stopping
        # If the validation loss improves, save the model and reset the counter. If not, increment the counter and check if it has reached the patience threshold.
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), 'best_resnet_model.pth')
            counter = 0
        else:
            counter += 1
            if counter >= patience:
                print("Early stopping triggered")
                break

if os.path.exists('best_resnet_model.pth'):
    model.load_state_dict(torch.load('best_resnet_model.pth'))
    print("Best Model Loaded")

#test evaluation
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for images, labels in test_loader:
        outputs = model(images)
        predicted = outputs.argmax(dim=1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

test_accuracy = correct / total * 100
print(f"Test Accuracy: {test_accuracy:.2f}%")

#Plotting training and validation loss
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].plot(train_losses, label='Train Loss')
axes[0].plot(val_losses,   label='Val Loss')
axes[0].set_title('Loss Curve')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Loss')
axes[0].legend()

axes[1].plot(val_accuracies, label='Val Accuracy', color='green')
axes[1].set_title('Validation Accuracy')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Accuracy (%)')
axes[1].legend()

plt.tight_layout()
plt.savefig('resnet_results.png')
plt.show()
print("Done.")
