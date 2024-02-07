# Import PyTorch
import torch
from torchvision import datasets, models, transforms
from torch import nn

# define the model
class CustomResnet50(nn.Module):
    def __init__(self, output_dim=1):
        super().__init__()
        # Load a pre-trained ResNet-50 model trained with the ImageNet dataset
        self.resnet = models.resnet50(pretrained=True)
        # Replace the final FC layer from the ResNet-50 model with a new FC layer
        # with the output dimension we want (in our case, 1 dimension for regression)
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, output_dim)

    def forward(self, x):
        # Permute the input to match the input shape of the ResNet-50 model
        x = x.permute(0, 3, 1, 2)
        # Pass the input through the ResNet-50 model
        return self.resnet(x)
    
if __name__ == '__main__':

    output_dim = 1 # regression problem
    model = CustomResnet50(output_dim)
    print(model)
