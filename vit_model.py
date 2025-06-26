import torch
from torchvision import models, transforms
from PIL import Image

class VisionTransformerClassifier:
    def __init__(self, device='cpu'):
        self.device = device
        # Load the pretrained Vision Transformer (ViT) model from torchvision
        self.model = models.vit_b_16(pretrained=True)
        self.model.eval()
        self.model.to(self.device)
        # Define the image transformation pipeline required by ViT
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], 
                std=[0.229, 0.224, 0.225]
            )
        ])

        # Load class labels for ImageNe
        with open('imagenet_classes.txt', 'r') as f:
            self.classes = [line.strip() for line in f.readlines()]
    def predict(self, image_path):
        image = Image.open(image_path).convert('RGB')
        input_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        # Perform inference without tracking gradients
        with torch.no_grad():
            output = self.model(input_tensor)
            probabilities = torch.nn.functional.softmax(output[0], dim=0)
            # Get the index of the class with highest probability
            confidence, predicted_idx = torch.max(probabilities, 0)
            predicted_class = self.classes[predicted_idx]
            return predicted_class, confidence.item()
