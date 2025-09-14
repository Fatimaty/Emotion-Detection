import torch
from PIL import Image
from torchvision import transforms
from model import EmotionCNN

def predict(image_path, weights="emotion_model.pth", classes=None, img_size=224, device="cpu"):
    model = EmotionCNN(num_classes=len(classes)).to(device)
    model.load_state_dict(torch.load(weights, map_location=device))
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    img = Image.open(image_path).convert("RGB")
    img_tensor = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(img_tensor)
        _, pred = torch.max(outputs, 1)

    return classes[pred]

if __name__ == "__main__":
    emotions = ["happy", "sad", "angry", "neutral"]
    label = predict("test.jpg", classes=emotions)
    print("Predicted emotion:", label)