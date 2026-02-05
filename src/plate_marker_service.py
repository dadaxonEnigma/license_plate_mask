import cv2
import numpy as np
from PIL import Image
import io
import logging
from django.conf import settings

from .plate_masker.inference import PlateDetector
from .plate_masker.geometry import order_points, is_bad_quad
from .plate_masker.blending import blur_region, prepare_plaque, warp_and_blend

logger = logging.getLogger(__name__)


class PlateMarkerService:
    
    _instance = None
    
    def __init__(self):
        self.detector = PlateDetector.get_instance()
        self.plaque_image = self._load_plaque()
    
    @classmethod
    def get_instance(cls):

        if cls._instance is None:
            cls._instance = cls()
        return cls._instance
    
    def _load_plaque(self):

        plaque_path = getattr(settings, 'YOLO_PLAQUE_PATH', None)
        if not plaque_path:
            logger.warning("YOLO_PLAQUE_PATH не установлен, плашка не будет использоваться")
            return None
        
        try:
            plaque = cv2.imread(plaque_path, cv2.IMREAD_UNCHANGED)
            if plaque is None:
                logger.error(f"Не удалось загрузить плашку из {plaque_path}")
                return None
                
            return prepare_plaque(plaque)
        except Exception as e:
            logger.error(f"Ошибка загрузки плашки: {e}")
            return None
    
    def process_image_file(self, image_path, output_path=None, conf=0.8):
  
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Не удалось загрузить изображение: {image_path}")
        
        result_img = self._process_cv2_image(img, conf)
        
        if output_path:
            cv2.imwrite(output_path, result_img)
            logger.info(f"Изображение сохранено: {output_path}")
        
        return result_img
    
    def process_pil_image(self, pil_image, conf=0.8):

        img_array = np.array(pil_image)
        
        if img_array.shape[2] == 4:
            img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2BGR)
        else:
            img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        
        result_array = self._process_cv2_image(img_array, conf)
        
        result_rgb = cv2.cvtColor(result_array, cv2.COLOR_BGR2RGB)
        return Image.fromarray(result_rgb)
    
    def _process_cv2_image(self, img, conf=0.8):

        result_img = img.copy()
        
        masks = self.detector.detect_masks_from_array(img, conf=conf)
        
        if not masks:
            logger.debug("Номера не найдены")
            return result_img
        
        found_any = False
        
        for i, poly in enumerate(masks):
            found_any = True
            try:
                poly = np.array(poly, dtype=np.float32)
                quad = order_points(poly)

                if is_bad_quad(quad):
                    logger.debug(f'Маска {i}: плохая геометрия — blur')
                    result_img = blur_region(result_img, poly)
                    continue
                    
                if self.plaque_image is not None:
                    try:
                        result_img = warp_and_blend(result_img, self.plaque_image, quad)
                    except Exception as e:
                        logger.warning(f'Плашка не смогла встать, применяем blur: {e}')
                        result_img = blur_region(result_img, poly)
                else:
                    result_img = blur_region(result_img, poly)

            except Exception as e:
                logger.error(f"Ошибка при обработке объекта {i}: {e}")
                continue
        
        return result_img
    
    def process_image_bytes(self, image_bytes, conf=0.8):
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            raise ValueError("Не удалось декодировать изображение из bytes")
        
        result_img = self._process_cv2_image(img, conf)
        
        _, buffer = cv2.imencode('.jpg', result_img)
        return buffer.tobytes()


def apply_plate_overlay(image: Image.Image) -> Image.Image:

    try:
        service = PlateMarkerService.get_instance()
        return service.process_pil_image(image)
    except Exception as e:
        logger.error(f"Ошибка в apply_plate_overlay: {e}")
        return image