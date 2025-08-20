import os
import cv2
import torch
import torch.nn as nn
import numpy as np
from torchvision.models import resnet50
from src.config import CLASSES

class Classifier:
    def __init__(self, model_path="trained_models/last.pt"):
        self.model_path = model_path
        self.model = resnet50()
        self.model.fc = nn.Linear(self.model.fc.in_features, len(CLASSES))
        
        # Load model trên CPU
        if self.model_path is not None and os.path.isfile(self.model_path):
            checkpoint = torch.load(self.model_path, map_location=torch.device("cpu"))
            self.model.load_state_dict(checkpoint["model_params"])
            print(f"[INFO] Đã tải mô hình ký tự từ: {self.model_path}")
        else:
            print(f"[ERROR] Không tìm thấy mô hình ký tự tại: {self.model_path}")
            exit(0)

        self.model.to(torch.device("cpu"))
        self.model.eval()

    def prediction(self, ori_image, draw=True):
        device = torch.device("cpu")
        # Lưu ảnh gốc để vẽ
        image_to_draw = ori_image.copy() if draw else None
        # Preprocessing
        image = cv2.resize(ori_image, (224, 224))
        image = np.transpose(image, (2, 0, 1)) / 255.0
        image = image[None, :, :, :]
        image = torch.from_numpy(image).float().to(device)

        # Prediction
        with torch.no_grad():
            results = self.model(image)
            probabilities = torch.softmax(results, dim=1).numpy()[0]
            prediction = np.argmax(probabilities)
            confidence = probabilities[prediction]

        if draw and image_to_draw is not None:
            cv2.putText(image_to_draw, f"{CLASSES[prediction]} ({confidence:.2f})", 
                       (50, 50), cv2.FONT_HERSHEY_COMPLEX, 2, (0, 255, 0), 2)
            return list(results[0]), prediction, image_to_draw, confidence

        return list(results[0]), prediction, ori_image, confidence