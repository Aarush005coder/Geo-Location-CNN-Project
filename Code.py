# ======================================== CELL - 1 ==========================================
import os
import cv2
import msgpack
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.layers import Dense, Conv2D, BatchNormalization, Dropout
from tensorflow.keras.layers import MaxPooling2D, GlobalAveragePooling2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.callbacks import EarlyStopping
# ============================================================================================

# ======================================== CELL - 2 ==========================================
print(os.listdir('/kaggle/input/datasets/habedi/large-dataset-of-geotagged-images'))
# ============================================================================================

# ======================================== CELL - 3 ==========================================
dataset_path = '/kaggle/input/datasets/habedi/large-dataset-of-geotagged-images'

for root, dirs, files in os.walk(dataset_path):
    print("ROOT:", root)
    print("DIRS:", dirs[:5])
    print("FILES:", files[:5])
    print("-"*50)
# ============================================================================================

# ======================================== CELL - 4 ==========================================
file_path = "/kaggle/input/datasets/habedi/large-dataset-of-geotagged-images/shards/shard_24.msg"

with open(file_path, "rb") as f:
    unpacker = msgpack.Unpacker(f, raw=False)

    for i, data in enumerate(unpacker):
        print(type(data))
        print(data)

        if i == 0:
            break
# ============================================================================================

# ======================================== CELL - 5 ==========================================
file_path = "/kaggle/input/datasets/habedi/large-dataset-of-geotagged-images/shards/shard_24.msg"

with open(file_path, "rb") as f:

    unpacker = msgpack.Unpacker(f, raw=False)

    first_item = next(unpacker)

print(first_item.keys())
# ============================================================================================

# ======================================== CELL - 6 ==========================================
with open(file_path, "rb") as f:
    unpacker = msgpack.Unpacker(f, raw=False)

    for i, item in enumerate(unpacker):
        image_bytes = item['image']
        image_array = np.frombuffer(image_bytes, np.uint8)
        
        image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        plt.figure(figsize=(4,4))
        plt.imshow(image)
        plt.axis('off')
        plt.show()

        if i == 4:
            break
# ============================================================================================

# ======================================== CELL - 7 ==========================================
import msgpack
import numpy as np
import cv2

images = []
targets = []
limit = 1000   

with open(file_path, "rb") as f:
    unpacker = msgpack.Unpacker(f, raw=False)

    for i, item in enumerate(unpacker):
        image_bytes = item['image']
        image_array = np.frombuffer(image_bytes, np.uint8)
        
        image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        image = cv2.resize(image, (128,128))
        image = image.astype('float32') / 255.0
        images.append(image)

        targets.append([
            item['latitude'],
            item['longitude']
        ])

        if i >= limit:
            break

X = np.array(images, dtype='float32')
y = np.array(targets, dtype='float32')

print(X.shape)
print(y.shape)
# ============================================================================================

# ======================================== CELL - 8 ==========================================
y[:,0] = (y[:,0] + 90) / 180
y[:,1] = (y[:,1] + 180) / 360
# ============================================================================================

# ======================================== CELL - 9 ==========================================
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)
# ============================================================================================

# ======================================== CELL - 10 =========================================
model = Sequential()

model.add(Conv2D(
    96,
    kernel_size=(11,11),
    strides=4,
    padding='same',
    activation='relu',
    input_shape=(128,128,3)
))

model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(3,3), strides=2))

model.add(Conv2D(
    256,
    kernel_size=(5,5),
    padding='same',
    activation='relu'
))

model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(3,3), strides=2))

model.add(Conv2D(
    384,
    kernel_size=(3,3),
    padding='same',
    activation='relu'
))

model.add(BatchNormalization())

model.add(Conv2D(
    384,
    kernel_size=(3,3),
    padding='same',
    activation='relu'
))

model.add(BatchNormalization())

model.add(Conv2D(
    256,
    kernel_size=(3,3),
    padding='same',
    activation='relu'
))

model.add(BatchNormalization())

model.add(GlobalAveragePooling2D())

model.add(Dense(512, activation='relu'))
model.add(Dropout(0.5))

model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))

model.add(Dense(2, activation='sigmoid'))
# ============================================================================================

# ======================================== CELL - 11 =========================================
model.summary()
# ============================================================================================

# ======================================== CELL - 12 =========================================
early_stop = EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True
)
# ============================================================================================

# ======================================== CELL - 13 =========================================
lr_scheduler = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=3,
    verbose=1
)
# ============================================================================================

# ======================================== CELL - 14 =========================================
model.compile(
    loss = 'mse',
    optimizer = 'adam',
    metrics = ['mae']
)
# ============================================================================================

# ======================================== CELL - 15 =========================================
history1 = model.fit(
    X_train,
    y_train,
    epochs=50,
    validation_data=(X_test, y_test),
    callbacks=[early_stop, lr_scheduler]
)
# ============================================================================================

# ======================================== CELL - 16 =========================================
plt.plot(history1.history['loss'], label='Loss')
plt.plot(history1.history['val_loss'], label='Validation Loss')
plt.xlabel('No. of epochs')
plt.ylabel('Loss')
plt.title('Loss v/s Validation Loss')
plt.legend()
plt.grid(True)
plt.show()
# ============================================================================================

# ======================================== CELL - 17 =========================================
plt.plot(history1.history['mae'], label='MAE')
plt.plot(history1.history['val_mae'], label='Validation MAE')

plt.xlabel('No. of epochs')
plt.ylabel('MAE')
plt.title('MAE vs Validation MAE')

plt.legend()
plt.grid(True)
plt.show()
# ============================================================================================

# ======================================== CELL - 18 =========================================
import cv2
import numpy as np
import matplotlib.pyplot as plt

image_path = "/kaggle/input/datasets/aarushkhandelwal/test-dataset/image28.jpeg"   

img = cv2.imread(image_path)

img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img_resized = cv2.resize(img_rgb, (128,128))

img_normalized = img_resized.astype('float32') / 255.0
img_input = np.expand_dims(img_normalized, axis=0)


prediction = model.predict(img_input)
pred_lat = prediction[0][0]
pred_lon = prediction[0][1]

pred_lat = pred_lat * 180 - 90
pred_lon = pred_lon * 360 - 180

plt.figure(figsize=(6,6))
plt.imshow(img_rgb)
plt.axis('off')
plt.title(f"Predicted Location\nLatitude: {pred_lat:.4f}, Longitude: {pred_lon:.4f}")
plt.show()

print("Predicted Latitude :", pred_lat)
print("Predicted Longitude:", pred_lon)
# ============================================================================================

# ======================================== CELL - 19 =========================================
import matplotlib.pyplot as plt

labels = ['Latitude', 'Longitude']
values = [pred_lat, pred_lon]

plt.figure(figsize=(6,5))
plt.bar(labels, values)
plt.xlabel('Coordinates')
plt.ylabel('Value')
plt.title('Predicted Coordinates')

for i, v in enumerate(values):
    plt.text(i, v, f'{v:.2f}', ha='center')

plt.grid(True)
plt.show()
# ============================================================================================

# ======================================== CELL - 20 =========================================
import math
)
def haversine_km(lat1, lon1, lat2, lon2):
    R    = 6371   
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a    = (math.sin(dlat/2)**2 +
            math.cos(math.radians(lat1)) *
            math.cos(math.radians(lat2)) *
            math.sin(dlon/2)**2)
    return R * 2 * math.asin(math.sqrt(a))


predictions = model.predict(X_test)
pred_lats = predictions[:, 0] * 180 - 90
pred_lons = predictions[:, 1] * 360 - 180

actual_lats = y_orig_test[:, 0]
actual_lons  = y_orig_test[:, 1]

print("\n" + "="*75)
print(f"{'#':<5} {'Actual Lat':>12} {'Actual Lon':>12} {'Pred Lat':>12} {'Pred Lon':>12} {'Error(km)':>12}")
print("="*75)

total_error = 0
for i in range(len(X_test)):
    a_lat = actual_lats[i]
    a_lon = actual_lons[i]
    p_lat = pred_lats[i]
    p_lon = pred_lons[i]
    error = haversine_km(a_lat, a_lon, p_lat, p_lon)
    total_error += error
    print(f"{i+1:<5} {a_lat:>12.4f} {a_lon:>12.4f} {p_lat:>12.4f} {p_lon:>12.4f} {error:>12.1f}")

avg_error = total_error / len(X_test)
print("="*75)
print(f"{'Average Error (km)':<50} {avg_error:>12.1f}")
print("="*75)

num_samples = 8
fig, axes   = plt.subplots(2, 4, figsize=(20, 10))
axes        = axes.flatten()

for i in range(num_samples):
    img = (X_test[i] * 255).astype(np.uint8)  

    a_lat = actual_lats[i]
    a_lon  = actual_lons[i]
    p_lat  = pred_lats[i]
    p_lon  = pred_lons[i]
    error  = haversine_km(a_lat, a_lon, p_lat, p_lon)

    axes[i].imshow(img)
    axes[i].axis('off')
    axes[i].set_title(
        f"Sample {i+1}\n"
        f"✅ Actual  → Lat: {a_lat:.2f}, Lon: {a_lon:.2f}\n"
        f"🔴 Predicted → Lat: {p_lat:.2f}, Lon: {p_lon:.2f}\n"
        f"📏 Error: {error:.1f} km",
        fontsize=9,
        loc='center'
    )

plt.suptitle("Actual vs Predicted GPS Coordinates — GeoVision",
             fontsize=14, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig('actual_vs_predicted_samples.png', bbox_inches='tight', dpi=150)
plt.show()

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# --- Latitude scatter ---
axes[0].scatter(actual_lats, pred_lats, alpha=0.5, color='steelblue', s=15)
axes[0].plot(
    [actual_lats.min(), actual_lats.max()],
    [actual_lats.min(), actual_lats.max()],
    'r--', linewidth=2, label='Perfect Prediction'
)
axes[0].set_xlabel('Actual Latitude',    fontsize=12)
axes[0].set_ylabel('Predicted Latitude', fontsize=12)
axes[0].set_title('Latitude — Actual vs Predicted', fontsize=13)
axes[0].legend()
axes[0].grid(True)

# --- Longitude scatter ---
axes[1].scatter(actual_lons, pred_lons, alpha=0.5, color='darkorange', s=15)
axes[1].plot(
    [actual_lons.min(), actual_lons.max()],
    [actual_lons.min(), actual_lons.max()],
    'r--', linewidth=2, label='Perfect Prediction'
)
axes[1].set_xlabel('Actual Longitude',    fontsize=12)
axes[1].set_ylabel('Predicted Longitude', fontsize=12)
axes[1].set_title('Longitude — Actual vs Predicted', fontsize=13)
axes[1].legend()
axes[1].grid(True)

plt.suptitle('Actual vs Predicted Coordinates — Scatter Plot',
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('scatter_actual_vs_predicted.png', bbox_inches='tight', dpi=150)
plt.show()

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].plot(history.history['loss'],     label='Train Loss')
axes[0].plot(history.history['val_loss'], label='Val Loss')
axes[0].set_title('Loss vs Validation Loss')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('MSE Loss')
axes[0].legend()
axes[0].grid(True)

axes[1].plot(history.history['mae'],     label='Train MAE')
axes[1].plot(history.history['val_mae'], label='Val MAE')
axes[1].set_title('MAE vs Validation MAE')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('MAE')
axes[1].legend()
axes[1].grid(True)

plt.tight_layout()
plt.savefig('training_curves.png', bbox_inches='tight', dpi=150)
plt.show()

print("\n✅ Done! Files saved:")
print("   → actual_vs_predicted_samples.png")
print("   → scatter_actual_vs_predicted.png")
print("   → training_curves.png")
# ============================================================================================
