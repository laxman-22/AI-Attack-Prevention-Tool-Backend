import torch
from torchvision import models
import torch.nn as nn

class FineTunedResNet50(nn.Module):
    def __init__(self):
        super(FineTunedResNet50, self).__init__()
        self.resnet = models.resnet50(weights='DEFAULT')

        for param in self.resnet.parameters():
            param.requires_grad = False
        
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, 1)
        
        self.attack_output = nn.Linear(self.resnet.fc.in_features, 5)

    def forward(self, x):
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)
        x = self.resnet.layer1(x)
        x = self.resnet.layer2(x)
        x = self.resnet.layer3(x)
        x = self.resnet.layer4(x)
        x = self.resnet.avgpool(x)
        
        x = torch.flatten(x, 1)
        
        binary_pred = torch.sigmoid(self.resnet.fc(x)).squeeze()
        
        attack_pred = self.attack_output(x)
        
        return binary_pred, attack_pred
    
def predict(img):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FineTunedResNet50().to(device)
    model.load_state_dict(torch.load('imp_binary_model.pth', map_location=device))
    model.eval()

    with torch.no_grad():
        binary_pred, attack_pred = model(img)

    binary_pred = binary_pred.item()
    attack_pred = attack_pred.argmax(dim=1).item()

    return binary_pred, attack_pred

