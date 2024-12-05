import torch
from torchvision import transforms, models
from PIL import Image
import torch.nn as nn
from image_attack import fgsm, preprocess_image
from model import FineTunedResNet50
import torchvision
from torchvision.models import ResNet34_Weights

# Set the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load and preprocess the image
img_path = 'goldfish.JPG'  # Replace with your image path
img = Image.open(img_path).convert("RGB")

img_tensor = preprocess_image(img)  # Ensure this function outputs a torch.Tensor

# Load the base ResNet model
model = torchvision.models.resnet34(weights=ResNet34_Weights.DEFAULT).to(device)
model.eval()

# Load the FineTunedResNet50 model
attack_model = FineTunedResNet50().to(device)
attack_model.load_state_dict(torch.load('imp_binary_model.pth'))
attack_model.eval()

# Perform FGSM attack
attacked = fgsm(model=model, images=img_tensor.to(device), label=torch.tensor([23]).to(device), epsilon=0.03)

# Preprocess the attacked image for prediction
attacked_img = attacked.squeeze().permute(1, 2, 0).cpu().detach().numpy()
attacked_img = torch.from_numpy(attacked_img).permute(2, 0, 1).unsqueeze(0).float().to(device)

# Make predictions
with torch.no_grad():
    binary_pred, attack_pred = attack_model(attacked_img)

# Convert predictions
binary_pred = torch.sigmoid(binary_pred).item()  # Convert tensor to scalar
attack_pred = attack_pred.argmax(dim=1).item()  # Get the index of the max value for attack type

# Output results
print(f"Binary prediction (0 = clean, 1 = attacked): {binary_pred}")
print(f"Attack prediction (0 = no_attack, 1 = deepfool, 2 = fgsm, 3 = pgd, 4 = cw): {attack_pred}")
