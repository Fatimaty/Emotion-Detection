class CNN(nn.Module):
    def __init__(self, num_classes = 7):
        super(CNN, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),  # Use padding=1 to maintain the spatial dimensions
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        
        # compute the flattened size after the convolutions and pooling
        self.flattened_size = 128 * 6 * 6 # varies according to input size and architecture
        
        # fully connected layers
        self.fc1 = nn.Linear(self.flattened_size, 128)
        self.fc2 = nn.Linear(128, num_classes)
        
    def forward(self, x):
        # apply the first two conv layers followed by pooling
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        
        # flatten the output for the dense layer
        x = x.view(x.size(0), -1)
        
        # apply the dense layers
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
model = CNN(num_classes = 4).to(device)
