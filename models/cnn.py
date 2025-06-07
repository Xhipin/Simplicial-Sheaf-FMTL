import torch
import torch.nn as nn

class SimpleCNN(nn.Module):
    """Simple CNN for MNIST (1 channel input)"""
    def __init__(self, num_classes=10):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2, 2)
        self.fc = nn.Linear(64 * 5 * 5, num_classes)
    
    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = x.view(-1, 64 * 5 * 5)
        x = self.fc(x)
        return x

class CIFAR10CNN(nn.Module):
    """CNN for CIFAR-10 (3 channel input)"""
    def __init__(self, num_classes=10):
        super(CIFAR10CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2, 2)
        self.fc = nn.Linear(64 * 6 * 6, num_classes)
    
    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = x.view(-1, 64 * 6 * 6)
        x = self.fc(x)
        return x

class HeterogeneousCNN:
    """Factory for creating CNNs of different sizes"""
    
    @staticmethod
    def small_cnn(input_channels=3, num_classes=10):
        """Small CNN with ~23K-28K parameters"""
        class SmallCNN(nn.Module):
            def __init__(self):
                super(SmallCNN, self).__init__()
                self.conv1 = nn.Conv2d(input_channels, 16, 3)
                self.relu = nn.ReLU()
                self.pool = nn.MaxPool2d(2, 2)
                self.fc = nn.Linear(16 * 14 * 14 if input_channels == 3 else 16 * 13 * 13, num_classes)
            
            def forward(self, x):
                x = self.pool(self.relu(self.conv1(x)))
                x = x.view(x.size(0), -1)
                x = self.fc(x)
                return x
        
        return SmallCNN()
    
    @staticmethod
    def medium_cnn(input_channels=3, num_classes=10):
        """Medium CNN with ~121K-154K parameters"""
        class MediumCNN(nn.Module):
            def __init__(self):
                super(MediumCNN, self).__init__()
                self.conv1 = nn.Conv2d(input_channels, 24, 3)
                self.relu1 = nn.ReLU()
                self.pool1 = nn.MaxPool2d(2, 2)
                self.conv2 = nn.Conv2d(24, 48, 3)
                self.relu2 = nn.ReLU()
                self.pool2 = nn.MaxPool2d(2, 2)
                
                if input_channels == 3:  # CIFAR-10
                    self.fc = nn.Linear(48 * 6 * 6, num_classes)
                else:  # MNIST
                    self.fc = nn.Linear(48 * 5 * 5, num_classes)
            
            def forward(self, x):
                x = self.pool1(self.relu1(self.conv1(x)))
                x = self.pool2(self.relu2(self.conv2(x)))
                x = x.view(x.size(0), -1)
                x = self.fc(x)
                return x
        
        return MediumCNN()
    
    @staticmethod
    def large_cnn(input_channels=3, num_classes=10):
        """Large CNN with ~212K-297K parameters"""
        class LargeCNN(nn.Module):
            def __init__(self):
                super(LargeCNN, self).__init__()
                self.conv1 = nn.Conv2d(input_channels, 32, 3)
                self.relu1 = nn.ReLU()
                self.pool1 = nn.MaxPool2d(2, 2)
                self.conv2 = nn.Conv2d(32, 64, 3)
                self.relu2 = nn.ReLU()
                self.pool2 = nn.MaxPool2d(2, 2)
                
                if input_channels == 3:  # CIFAR-10
                    self.fc1 = nn.Linear(64 * 6 * 6, 120)
                else:  # MNIST
                    self.fc1 = nn.Linear(64 * 5 * 5, 120)
                
                self.relu3 = nn.ReLU()
                self.fc2 = nn.Linear(120, num_classes)
            
            def forward(self, x):
                x = self.pool1(self.relu1(self.conv1(x)))
                x = self.pool2(self.relu2(self.conv2(x)))
                x = x.view(x.size(0), -1)
                x = self.relu3(self.fc1(x))
                x = self.fc2(x)
                return x
        
        return LargeCNN()
