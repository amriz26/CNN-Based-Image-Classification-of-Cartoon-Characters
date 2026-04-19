import torch
import torch.nn as nn

class CartoonCNN(nn.Module):
    def __init__(self, num_classes):
        super(CartoonCNN, self).__init__()

        #extract features
        self.features = nn.Sequential(
            
            #block 1
            torch.nn.Conv2d(3,32,3),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),

            #block 2
            torch.nn.Conv2d(32,64,3),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),

            #block 3
            torch.nn.Conv2d(64,128,3),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),                     
        )

        #Classifier
        self.classifier = nn.Sequential(
            #flattening the 3d vector to 1d
            torch.nn.Flatten(),
            #final class output
            torch.nn.Linear(32768, num_classes)
        )

    def forward(self,x):
        x = self.features(x)
        x = self.classifier(x)
        return x