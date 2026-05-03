# 🌍📍GeoVision — Image-Based Geo-Location Predictor
A deep learning model that predicts the 'latitude and longitude' of any image using a CNN inspired by the legendary AlexNet architecture.

### What This Project Does
GeoVision takes a 'photo as input' and predicts where on Earth it was taken'.
The model learns visual patterns from thousands of geotagged images — things like landscape type, vegetation, architecture style, sky color, and lighting — and maps those patterns to real-world GPS coordinates (latitude & longitude).
It is a 'regression problem', not classification. Instead of predicting a category like "Paris" or "India", the model outputs two continuous numbers: `latitude` and `longitude`.

### Simple pipeline
                                                          Input Image
                                                              ↓
                                                Resize to 128×128 + Normalize pixels
                                                              ↓
                                                      AlexNet-style CNN
                                                              ↓
                                                  Predicted Latitude & Longitude

## Model Architecture — AlexNet Inspired
This project draws direct inspiration from 'AlexNet (Krizhevsky et al., 2012)' — the landmark CNN that won ImageNet 2012 and changed computer vision forever.

### Side-by-Side Comparison
|       Layer       |        AlexNet (Original)     |          GeoVision (This Project)       |
|-------------------|-------------------------------|-----------------------------------------|
| Conv Layer 1      | `Conv(96, 11×11, stride=4)`   | ✅ Identical                            |
| Conv Layer 2      | `Conv(256, 5×5)`              | ✅ Identical                            |
| Conv Layer 3      | `Conv(384, 3×3)`              | ✅ Identical                            |
| Conv Layer 4      | `Conv(384, 3×3)`              | ✅ Identical                            |
| Conv Layer 5      | `Conv(256, 3×3)`              | ✅ Identical                            |
| Pooling           | MaxPooling after layers 1 & 2 | ✅ Identical                            |
| Regularization    | Dropout(0.5)                  | ✅ Identical                            |
| Normalization     | Local Response Norm (LRN)     | BatchNormalization (modern upgrade)      |
| Fully Connected   | `4096 → 4096 → 1000`          | `512 → 256 → 2` (adapted for regression) |
| Output Activation | Softmax (classification)      | Sigmoid (regression to [0,1])            |

### Key Adaptation
AlexNet was built for 1000-class image classification. GeoVision adapts it for coordinate regression by:
- Replacing `softmax` with `sigmoid` in the output layer
- Outputting only 2 neurons (lat, lon) instead of 1000
- Normalizing coordinates to `[0, 1]` range to match sigmoid output
- Using `GlobalAveragePooling2D` instead of Flatten to reduce parameters

### Architecture Diagram
                                                                    Input (128×128×3)
                                                                            │
                                                                    ┌───────▼────────┐
                                                                    │ Conv2D 96      │  11×11, stride 4
                                                                    │ BatchNorm      │
                                                                    │ MaxPooling     │  3×3, stride 2
                                                                    └───────┬────────┘
                                                                            │
                                                                    ┌───────▼────────┐
                                                                    │ Conv2D 256     │  5×5
                                                                    │ BatchNorm      │
                                                                    │ MaxPooling     │  3×3, stride 2
                                                                    └───────┬────────┘
                                                                            │
                                                                    ┌───────▼────────┐
                                                                    │ Conv2D 384     │  3×3
                                                                    │ BatchNorm      │
                                                                    └───────┬────────┘
                                                                            │
                                                                    ┌───────▼────────┐
                                                                    │ Conv2D 384     │  3×3
                                                                    │ BatchNorm      │
                                                                    └───────┬────────┘
                                                                            │
                                                                    ┌───────▼────────┐
                                                                    │ Conv2D 256     │  3×3
                                                                    │ BatchNorm      │
                                                                    │ GlobalAvgPool  │
                                                                    └───────┬────────┘
                                                                            │
                                                                    ┌───────▼────────┐
                                                                    │ Dense 512      │  ReLU
                                                                    │ Dropout 0.5    │
                                                                    └───────┬────────┘
                                                                            │
                                                                    ┌───────▼────────┐
                                                                    │ Dense 256      │  ReLU
                                                                    │ Dropout 0.5    │
                                                                    └───────┬────────┘
                                                                            │
                                                                    ┌───────▼────────┐
                                                                    │ Dense 2        │  Sigmoid
                                                                    │ (lat, lon)     │
                                                                    └────────────────┘


## Dataset
|       Property       |                                                     Details                                                                      |
|----------------------|----------------------------------------------------------------------------------------------------------------------------------|
| Source               | [`habedi/large-dataset-of-geotagged-images`](https://www.kaggle.com/datasets/habedi/large-dataset-of-geotagged-images) on Kaggle |
| Format               | MessagePack `.msg` shard files                                                                                                   |
| Each record contains | JPEG image bytes + latitude + longitude                                                                                          |
| Images used          | 1001 images from `shard_24.msg`                                                                                                  |
| Image size           | Resized to `128 × 128` pixels                                                                                                    |

### Preprocessing Steps
```python
# 1. Decode image from bytes
image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

# 2. Resize
image = cv2.resize(image, (128, 128))

# 3. Normalize pixels to [0, 1]
image = image.astype('float32') / 255.0

# 4. Normalize coordinates to [0, 1]
latitude  = (latitude  + 90)  / 180   # [-90,  90]  → [0, 1]
longitude = (longitude + 180) / 360   # [-180, 180] → [0, 1]
```

## Training Details

|     Parameter      |           Value                |
|--------------------|--------------------------------|
| Loss Function      | Mean Squared Error (MSE)       |
| Optimizer          | Adam                           |
| Learning Rate      | 0.001 (adaptive)               |
| Metrics            | Mean Absolute Error (MAE)      |
| Epochs             | Up to 50 (with Early Stopping) |
| Batch Size         | 32 (default)                   |
| Train / Test Split | 80% / 20%                      |
| Training Samples   | ~800                           |
| Validation Samples | ~200                           |

### Callbacks Used
```python
# Stops training when val_loss stops improving
EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Halves learning rate when val_loss plateaus
ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3)
```

## Learning Rate Schedule (from training logs)

| Epoch | Learning Rate |             Event                 |
|-------|---------------|-----------------------------------|
| 1–6   | 0.001         | Normal training                   |
| 7     | 0.0005        | ReduceLROnPlateau triggered       |
| 12    | 0.00025       | ReduceLROnPlateau triggered again |

## Results

### Training Log (Sample)

| Epoch | Train Loss | Val Loss | Train MAE | Val MAE |
|-------|------------|----------|-----------|---------|
| 1     | 0.0587     | 0.1993   | 0.1900    | 0.3965  |
| 3     | 0.0372     | 0.0407   | 0.1455    | 0.1476  |
| 4     | 0.0352     | 0.0377   | 0.1413    | 0.1376  |
| 9     | 0.0294     | 0.0367   | 0.1300    | 0.1482  |
| 13    | 0.0265     | 0.0368   | 0.1233    | 0.1410  |

### What the Numbers Mean
Val MAE ≈ 0.14 on normalized scale
→ Latitude  error ≈ 0.14 × 180° ≈ 25°  (~2,800 km)
→ Longitude error ≈ 0.14 × 360° ≈ 50°  (~5,500 km at equator)

### Observations
- The model learns quickly — val_loss drops from 0.199 → 0.037 in just 4 epochs
- Validation loss plateaus around 0.036 – 0.037 — a sign that more data is needed, not a bigger model
- Training loss keeps decreasing while val_loss plateaus — mild overfitting due to small dataset (1001 images)
- The LR scheduler fired twice, confirming the model got stuck in a flat region

### Future Improvements
|                       Improvement                     |                    Expected Impact                      |
|-------------------------------------------------------|---------------------------------------------------------|
| Load 5–10 shards (~5000–10000 images)                 | Biggest improvement — more data = better generalization |
| Use pretrained EfficientNetB0 or MobileNetV2 backbone | Transfer learning from ImageNet features                |
| Add data augmentation (flips, rotations, zoom)        | Reduces overfitting                                     |
| Use Haversine distance as loss/metric                 | More meaningful error in km instead of normalized MAE   |
| Train separate models for lat and lon                 | Decouple coordinate prediction                          |
| Build a web app with map visualization                | Show predicted location on an interactive map           |

## Folder Structure
```
GeoVision/
│
├── geovision.ipynb          # Main Kaggle notebook (full pipeline)
├── README.md                # Project documentation (this file)
│
├── sample_outputs/          # Example predictions
│   ├── predicted_image1.png
│   └── predicted_image2.png
│
└── models/                  # Saved model weights (optional)
    └── geovision_model.h5
```

## Tech Stack
|                                                    Tool                                                  |             Purpose            |
|----------------------------------------------------------------------------------------------------------|--------------------------------|
| ![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white)             | Core language                  |
| ![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=flat&logo=tensorflow&logoColor=white) | Model building & training      |
| ![Keras](https://img.shields.io/badge/Keras-D00000?style=flat&logo=keras&logoColor=white)                | High-level neural network API  |
| ![OpenCV](https://img.shields.io/badge/OpenCV-5C3EE8?style=flat&logo=opencv&logoColor=white)             | Image decoding & preprocessing |
| ![NumPy](https://img.shields.io/badge/NumPy-013243?style=flat&logo=numpy&logoColor=white)                | Array operations               |
| ![Matplotlib](https://img.shields.io/badge/Matplotlib-11557C?style=flat)                                 | Training curve visualization   |
| `msgpack`                                                                                                | Reading `.msg` shard files     |
| `scikit-learn`                                                                                           | Train/test split               |
| ![Kaggle](https://img.shields.io/badge/Kaggle-20BEFF?style=flat&logo=kaggle&logoColor=white)             | Dataset & compute platform     |


## Author

### Developed By Aarush Khandelwal

[![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=flat&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/aarush-khandelwal-1b99a7320/)
[![GitHub](https://img.shields.io/badge/GitHub-181717?style=flat&logo=github&logoColor=white)](https://github.com/Aarush005coder)
[![Kaggle](https://img.shields.io/badge/Kaggle-20BEFF?style=flat&logo=kaggle&logoColor=white)](https://www.kaggle.com/code/aarushkhandelwal/alexnet-architecture/edit)

## License
This project is open source and available under the [MIT License](LICENSE).
