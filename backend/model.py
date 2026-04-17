# Модуль загрузки и инференса модели детекции переломов

import torch
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image
import io
import base64
import os

class FractureDetector:
    def __init__(self, model_path: str = "best.pt", device: str = None):
        '''
        В данном методе инициализируется модель
        
        Args:
                model_path: путь к файлу модели .pt
                device: 'cuda' или 'cpu' 
                '''
    
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Модель не найдена: {model_path}")
        
        self.model = YOLO(model_path)
        self.model.to(device)
        self.conf_threshold = 0.25
        self.class_names = self.model.names
        
        
    def predict_from_bytes(self, image_bytes: bytes) -> dict:
        '''
        Данный метод реализует предсказания на основе байтов изображения
        
        Args:
            image_bytes: байты изображения
        
        Returns:
            dict: результаты детекции
        '''
        
        image = Image.open(io.BytesIO(image_bytes))
        if image.mode != 'RGB':
            image = image.convert('RGB')  
        size = image.size
        
        results = self.model(image, conf=self.conf_threshold, device=self.device)
        
        #Парсинг результата (модель выдает bounding boxes, считается уверенность, затем на ее основе и класс)
        
        detections = []
        if len(results) > 0 and results[0].boxes is not None:
            boxes = results[0].boxes
            
            for i in range(len(boxes)):
                
                x1, y1, x2, y2 = boxes.xyxy[i].tolist()
                confidence = float(boxes.conf[i])
                class_id = int(boxes.cls[i])
                class_name = self.class_names.get(class_id, f"class_{class_id}")
                
                detections.append({
                    "bbox": [int(x1), int(y1), int(x2), int(y2)],
                    "confidence": round(confidence, 4),
                    "class_id": class_id,
                    "class_name": class_name
                })
        
    
        annotated_image = results[0].plot()
        annotated_image_rgb = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
        annotated_image_pil = Image.fromarray(annotated_image_rgb)
        img_byte_arr = io.BytesIO()
        annotated_image_pil.save(img_byte_arr, format='PNG')
        annotated_image_base64 = base64.b64encode(img_byte_arr.getvalue()).decode('utf-8')
        
        return {
            "success": True,
            "num_detections": len(detections),
            "detections": detections,
            "image_size": size,
            "annotated_image_base64": annotated_image_base64
        }
        
    
    def predict_from_path(self, image_path: str) -> dict:
        """
        Данный метод реализует редсказание на основе пути к изображению
        
        Args:
            image_path: путь к изображению
        
        Returns:
            dict: результаты детекции
        """
        with open(image_path, 'rb') as f:
            image_bytes = f.read()
        
        return self.predict_from_bytes(image_bytes)
    
    
    
_detector = None
def get_detector(model_path: str = "best.pt"):
    """Получение глобального экземпляра детектора"""
    global _detector
    if _detector is None:
        _detector = FractureDetector(model_path=model_path)
    return _detector