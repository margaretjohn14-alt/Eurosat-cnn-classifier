import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
import numpy as np
import os

#Transforms for the dataset
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.3444, 0.3803, 0.4078],
        std=[0.1995, 0.1348, 0.1138]
    )
])

#Load the dataset
dataset = ImageFolder(root=r'C:\Users\john_mg\U-net_segmentation\EuroSAT\2750',
                      transform=transform)

#Split the dataset into training, validation and test sets
total = len(dataset)
train_size = int(0.7 * total) #70% for training
val_size = int(0.15 * total) #15% for validation
test_size = total - train_size - val_size #15% for testing

train_set, val_set, test_set = random_split(dataset, [train_size, val_size, test_size])

#Create DataLoaders for each sets
train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
val_loader = DataLoader(val_set, batch_size=32, shuffle=False)
test_loader = DataLoader(test_set, batch_size=32, shuffle=False)

#Verify the splits
print(f"Total images: {total}")
print(f"Training images: {len(train_set)}")
print(f"Validation images: {len(val_set)}")
print(f"Testing images: {len(test_set)}")

#Check one batch
images, labels = next(iter(train_loader))
print(f"\nBatch image shape:{images.shape}")
print(f"\nBatch label shape:{labels.shape}")

#Build the CNN model
class EuroSATClassifier(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()

        #Feature extraction- 3 convolutional layers with ReLU and max pooling- learns WHAT is in the image
        self.features = nn.Sequential(
            #BLOCK 1
            nn.Conv2d(3, 32, kernel_size=3, padding=1), #in --> 3, out --> 32
            nn.ReLU(),
            nn.MaxPool2d(2), #64x64 -->32x32
            
            #BLOCK 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2), #32x32 -->16x16
            
            #BLOCK 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2), #16x16 -->8x8
        )

        #Classifier - decides which class the image blongs to based on the features extracted by the convolutional layers
        self.classifier = nn.Sequential(
            nn.Flatten(), #flatten the 3D feature maps into a 1D vector (128x8x8 = 8192) to feed into the fully connected layers
            nn.Linear(128 * 8 * 8, 256), # fully connected layer that takes the flattened feature maps and outputs a vector of size 256
            nn.ReLU(),
            nn.Dropout(0.5), #dropout layer to prevent overfitting by randomly setting 50% of the activations to zero during training
            nn.Linear(256, num_classes) #final output layer that maps the 256-dimensional vector to the number of classes (10 in this case)
        )
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x
    
#Verify the model architecture
model = EuroSATClassifier()
print(model)

#test with one batch
with torch.no_grad():
    output = model(images)
    print(f"\nOutput images: {output.shape}")  #should be (batch_size, num_classes) --> (32, 10) 

# Training setup
criterion = nn.CrossEntropyLoss() #loss function for multi-class classification
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

train_losses = []
val_losses = []
val_accuracies = []

EPOCHS = 15
#  Early Stopping Setup 
best_val_loss = float('inf')
patience      = 3
counter       = 0

TRAIN = True #change to true to train the model, false to load the best model and test it on the test set
if TRAIN:
    #Training Loop
    for epoch in range(EPOCHS):
        #Training
        model.train()
        total_train_loss = 0
        for images, labels in train_loader:
            optimizer.zero_grad() #clear previous gradients
            outputs = model(images) #forward pass
            loss = criterion(outputs, labels) #calculate loss
            loss.backward() #backpropagation
            optimizer.step() #update weights
            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        #Validation
        model.eval()
        correct = 0 #Counts how many predictions were right
        total = 0 #counts how many images were processed
        total_val_loss = 0
        with torch.no_grad():
            for images, labels in val_loader:
                outputs = model(images)
                loss = criterion(outputs, labels) #calculate validation loss
                total_val_loss += loss.item()

                predicted = outputs.argmax(dim=1) #get the index of the class with the highest score
                correct += (predicted == labels).sum().item() #count correct predictions
                total += labels.size(0) #count total images

        avg_val_loss = total_val_loss / len(val_loader)
        val_accuracy = correct/total * 100
        val_losses.append(avg_val_loss)
        val_accuracies.append(val_accuracy)

        print(f"Epoch: {epoch+1}/{EPOCHS}, | Train Loss: {avg_train_loss:.4f}, | Val Loss: {avg_val_loss:.4f}, | Val Accuracy: {val_accuracy:.2f}%")

        #Early stopping check
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            counter       = 0
            torch.save(model.state_dict(), 'best_model.pth')
            print(f"New best model saved (val loss: {best_val_loss:.4f})")
        else:
            counter += 1
            print(f"No improvement ({counter}/{patience})")
            if counter >= patience:
                print(f"\nEarly stopping at epoch {epoch+1}")
                break

if os.path.exists('best_model.pth'):
    model.load_state_dict(torch.load('best_model.pth'))
    print("Best model loaded.")
else:
    print("No saved model. Set TRAIN = True first.")

#Testing
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for images, labels in test_loader:
        outputs = model(images)
        predicted = outputs.argmax(dim=1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

test_accuracy = correct/total * 100
print(f"Test Accuracy: {test_accuracy:.2f}%")

#plot training and validation loss
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].plot(train_losses, label="Train Loss")
axes[0].plot(val_losses, label="Validation Loss")
axes[0].set_title("Loss Curve")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Loss")
axes[0].legend()

axes[1].plot(val_accuracies, label="Validation Accuracy", color='green')
axes[1].set_title("Valiadation Accuracy Curve")
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("Accuracy (%)")
axes[1].legend()

plt.tight_layout()
plt.savefig("training_results.png")
plt.show()
print("Done.")

