import tensorflow as tf
import os
import time


# Dataset de rotaciones predefinidas (0°, 90°, 180°, 270°)
def preprocess_image_tf(file_path, image_size=224):
    image = tf.io.read_file(file_path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [image_size, image_size])

    # Selección aleatoria de rotación: 0 (0°), 1 (90°), 2 (180°), 3 (270°)
    k = tf.random.uniform([], minval=0, maxval=4, dtype=tf.int32)
    label = k
    image = tf.image.rot90(image, k=k)  # Rotar la imagen 90° k veces
    image = (image / 127.5) - 1.0  # Normalizar a [-1, 1]

    return image, label


def create_rotnet_dataset(image_dir, image_size=224, batch_size=32):
    print(f"📁 Cargando imágenes desde: {image_dir}")
    start_time = time.time()
    image_paths = [
        os.path.join(image_dir, fname)
        for fname in os.listdir(image_dir)
        if fname.lower().endswith((".jpg", ".jpeg", ".png"))
    ]

    dataset = tf.data.Dataset.from_tensor_slices(image_paths)
    dataset = dataset.map(preprocess_image_tf, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.shuffle(1000)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    elapsed = time.time() - start_time
    print(f"⏱️ Preprocesamiento completado en {elapsed:.2f} segundos.\n")
    # save
    return dataset


# Modelo simple CNN (puedes cambiarlo por ResNet, etc.)
def build_rotnet_model(input_shape=(224, 224, 3), num_classes=4):
    print("🔧 Construyendo el modelo RotNet...")
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Conv2D(32, 3, activation="relu", input_shape=input_shape),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Conv2D(64, 3, activation="relu"),
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.Dense(num_classes, activation="softmax"),
        ]
    )
    return model


# === USO ===
# Cambia esta ruta por tu carpeta de imágenes reales
image_dir = "E:/TFM/pocRoten"  # <- ajusta a tu dataset
# image_dir = "E:/TFM/PlantsClassification/test/cucumber"
dataset = create_rotnet_dataset(image_dir)

model = build_rotnet_model()
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
print("✅ Modelo creado con éxito.\n")
print("🚀 Comenzando el entrenamiento...\n")


history = model.fit(dataset, epochs=5, verbose=1)


model.save("E:/TFM/rotnet_model.keras")
print(f"\n✅ Entrenamiento finalizado en.")

save_path = "E:/TFM/rotnet_model.keras"
model.save(save_path)
print(f"💾 Modelo guardado en: {save_path}")
