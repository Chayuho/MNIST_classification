
import torch.nn as nn
import torch
import torch.nn.functional as F

RANDOM_SEED = 42

class LeNet5(nn.Module):
    """ LeNet-5 (LeCun et al., 1998)

        - For a detailed architecture, refer to the lecture note
        - Freely choose activation functions as you want
        - For subsampling, use max pooling with kernel_size = (2,2)
        - Output should be a logit vector
    """

    def __init__(self, num_class):
        super(LeNet5, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1)
        self.nonlinear = nn.Tanh()
        self.dropout = nn.Dropout(0.3)
        self.bn1 = nn.BatchNorm2d(6)
        self.bn2 = nn.BatchNorm2d(16)
        
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_class)
        

    def forward(self, img):
        output = self.conv1(img)
        output = self.bn1(output)
        output = self.nonlinear(output)
        output = self.pool(output)
        
        output = self.conv2(output)
        output = self.bn2(output)
        output = self.nonlinear(output)
        output = self.pool(output)

        output = output.view(-1, 16 * 5 * 5)

        output = self.fc1(output)
        output = self.nonlinear(output)
        output = self.dropout(output)
        
        output = self.fc2(output)
        output = self.nonlinear(output)
        output = self.dropout(output)
        output = self.fc3(output)

        return output


class CustomMLP(nn.Module):
    """ Your custom MLP model

        - Note that the number of model parameters should be about the same
          with LeNet-5
    """

    def __init__(self, input_size, num_class):
        super(CustomMLP, self).__init__()
        self.layer1 = nn.Linear(input_size, 59)
        self.layer2 = nn.Linear(59, 18)
        self.layer3 = nn.Linear(18, num_class)
        self.nonlinear = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        self.b_normal = nn.BatchNorm2d(59)
        

    def forward(self, img):
        output = self.layer1(img)
        output = self.b_normal(output)
        output = self.nonlinear(output)
        output = self.dropout(output)
        output = self.layer2(output)
        output = self.nonlinear(output)
        output = self.dropout(output)
        output = self.layer3(output)
        output = output.squeeze(dim=1)
        

        return output

if __name__ == '__main__':
    model = LeNet5(num_class=10)
    print("LeNet5 :", sum(p.numel() for p in model.parameters() if p.requires_grad))
    
    c_model = CustomMLP(1024, num_class=10)
    print("CustomMLP :", sum(p.numel() for p in c_model.parameters() if p.requires_grad))
    
    
