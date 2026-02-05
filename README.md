# Plate Detection & Masking

Система автоматической детекции и маскирования номерных знаков на изображениях с использованием YOLOv8 Segmentation.

## Возможности

- Детекция номерных знаков на изображениях
- Два режима маскирования:
  - Размытие номера (Gaussian blur)
  - Наложение плашки поверх номера
- Пакетная обработка изображений
- Поддержка GPU (CUDA) и CPU
- Django-интеграция для веб-сервисов

## Установка

### 1. Клонировать репозиторий

```bash
git clone <your-repo-url>
cd plate_det
```

### 2. Установить зависимости

```bash
pip install -r requirements.txt
```

### 3. Скачать модель

Обученная модель находится в:
```
notebooks/runs/segment/train/weights/best.pt
```

## Структура проекта

```
plate_det/
├── src/
│   ├── plate_masker/          # Основной модуль детекции и маскирования
│   │   ├── inference.py       # Класс PlateDetector для детекции
│   │   ├── geometry.py        # Геометрические утилиты
│   │   └── blending.py        # Маскирование и размытие
│   ├── test_inference.py      # Пакетная обработка изображений
│   ├── build_dataset.py       # Подготовка датасета для обучения
│   └── plate_marker_service.py # Django-сервис
├── notebooks/
│   ├── model.ipynb            # Обучение модели
│   └── test.ipynb             # Быстрое тестирование
├── test_images/               # Тестовые изображения (~600 фото)
├── dataset/                   # Датасет для обучения (575 изображений)
├── assets/autouz.png          # Плашка для замены номера
└── utils/test_model.py        # Оценка производительности модели
```

## Использование

### Быстрый тест (один файл)

```python
from src.plate_masker.inference import PlateDetector
from src.plate_masker.blending import blur_region, prepare_plaque, warp_and_blend
from src.plate_masker.geometry import order_points, is_bad_quad
import cv2

# Загрузить модель
detector = PlateDetector(
    model_path='notebooks/runs/segment/train/weights/best.pt',
    device='auto'  # или 'cuda:0', 'cpu'
)

# Детектировать номера
image_path = 'test_images/car1.jpg'
masks = detector.detect_masks(image_path, conf=0.8, imgsz=1024)

if masks:
    print(f"Найдено номеров: {len(masks)}")
else:
    print("Номера не найдены")
```

### Пакетная обработка

Обработать папку с изображениями:

```bash
cd src
python test_inference.py
```

**Настройка в коде:**
- `input_folder` - папка с исходными изображениями
- `output_base` - папка для результатов
- `conf_threshold` - порог уверенности (по умолчанию 0.8)

**Результаты сохраняются в:**
- `outputs2/processed/` - изображения с найденными номерами (замаскированы)
- `outputs2/no_number/` - изображения без номеров

### Оценка модели

Получить статистику детекции на тестовом наборе:

```bash
cd utils
python test_model.py
```

Показывает:
- Процент изображений с детекцией
- Среднюю уверенность (confidence)
- Площадь bounding boxes

## Тестирование

### Тест 1: Базовая детекция

```bash
cd utils
python test_model.py
```

**Ожидается:**
- Detection rate > 80%
- Average confidence > 0.7
- Без ошибок при загрузке модели

### Тест 2: Пакетная обработка

```bash
cd src
python test_inference.py
```

**Проверить:**
1. Создаются папки `outputs2/processed/` и `outputs2/no_number/`
2. Изображения корректно распределены по папкам
3. На обработанных изображениях номера замаскированы
4. Нет ошибок при обработке различных форматов (jpg, png, webp)

### Тест 3: Интерактивный тест (Jupyter)

```bash
jupyter notebook notebooks/test.ipynb
```

**Проверить:**
- Визуализацию детекций
- Корректность масок на различных ракурсах
- Качество размытия/наложения плашки

### Тест 4: Django-сервис (если используется)

```python
from src.plate_marker_service import PlateMarkerService

service = PlateMarkerService()
result = service.process_image_file('test_images/car1.jpg')
# result содержит обработанное изображение с замаскированными номерами
```

## Параметры модели

**Обученная модель (best.pt):**
- Архитектура: YOLOv8s-seg (segmentation)
- Датасет: 575 размеченных изображений
- Train/Val split: 85% / 15%
- Эпох обучения: 100 (early stopping: patience=30)
- Размер входа: 1024x1024
- Оптимизатор: AdamW
- Learning rate: 0.0005

**Параметры инференса:**
- `conf`: порог уверенности (рекомендуется 0.8)
- `imgsz`: размер изображения для модели (рекомендуется 1024)
- `device`: 'cuda:0' (GPU) или 'cpu'

## Известные ограничения

- Модель оптимизирована для узбекских номерных знаков
- Детекция может быть хуже на сильно размытых или малых номерах (< 50px)
- При плохой геометрии полигона (is_bad_quad) применяется размытие вместо плашки
- Django-интеграция требует настройки settings.YOLO_MODEL_PATH

## Производительность

- **GPU (CUDA)**: ~50-100 FPS (зависит от GPU)
- **CPU**: ~5-15 FPS
- **Размер модели**: 23 MB

## Лицензия

[Добавьте информацию о лицензии]

## Авторы

[Добавьте информацию об авторах]
