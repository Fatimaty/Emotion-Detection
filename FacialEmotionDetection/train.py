import torch
import torch.nn as nn
import torch.optim as optim
from model import EmotionCNN
from utils import get_dataloaders

def train(data_dir, epochs=10, lr=0.001, batch_size=32, device="cuda"):
    train_loader, val_loader, classes = get_dataloaders(data_dir, batch_size=batch_size)
    model = EmotionCNN(num_classes=len(classes)).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(train_loader):.4f}")

    torch.save(model.state_dict(), "emotion_model.pth")
    print("Model saved as emotion_model.pth")

if __name__ == "__main__":
    train(data_dir="KDEF", epochs=20, lr=0.001, batch_size=32, device="cuda")