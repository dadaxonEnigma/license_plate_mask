"""
Инференс модели YOLO для детекции номерных знаков
"""

import torch
from ultralytics import YOLO
from django.conf import settings
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class PlateDetector:
    """
    Класс для детекции номерных знаков с использованием YOLO
    """
    
    _instance = None
    
    def __init__(self, model_path=None, device='auto'):
        self.model_path = model_path or getattr(settings, 'YOLO_MODEL_PATH', None)
        if not self.model_path:
            raise ValueError("YOLO_MODEL_PATH не установлен в settings")
        
        if device == 'auto':
            self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
            
        self.model = None
        self._load_model()
    
    @classmethod
    def get_instance(cls):
        """
        Получение экземпляра детектора (синглтон)
        """
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance
    
    def _load_model(self):
        """Загружает модель YOLO"""
        try:
            logger.info(f"Загрузка модели YOLO из {self.model_path} на {self.device}")
            self.model = YOLO(self.model_path)
            logger.info("Модель YOLO успешно загружена")
            
        except Exception as e:
            logger.error(f"Ошибка загрузки модели YOLO: {e}")
            raise
    
    def detect_masks(self, image_path, conf=0.8, imgsz=1024):
        """
        Детектирует маски номерных знаков на изображении
        
        Возвращает:
            list: Список полигонов масок или пустой список если номера не найдены
        """
        try:
            pred = self.model.predict(
                image_path,
                imgsz=imgsz,
                conf=conf,
                verbose=False,
                device=self.device
            )[0]

            if pred.masks is None or len(pred.masks.xy) == 0:
                return []

            return pred.masks.xy
            
        except Exception as e:
            logger.error(f"Ошибка детекции на изображении {image_path}: {e}")
            return []
    
    def detect_masks_from_array(self, img_array, conf=0.8, imgsz=1024):
        """
        Детектирует маски номерных знаков из numpy array
        """
        try:
            pred = self.model.predict(
                img_array,
                imgsz=imgsz,
                conf=conf,
                verbose=False,
                device=self.device
            )[0]

            if pred.masks is None or len(pred.masks.xy) == 0:
                return []

            return pred.masks.xy
            
        except Exception as e:
            logger.error(f"Ошибка детекции из массива: {e}")
            return []