import torch
from .model import VisionModel

def run_inference(image_tensor):
    model = VisionModel()
    model.eval()
    with torch.no_grad():
        predictions = model(image_tensor)
    return predictions
