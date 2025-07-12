import tensorflow as tf
import os
import random

# Etiquetas y rotaciones disponibles
ROTATION_LABELS = {0: 0, 90: 1, 180: 2, 270: 3}
ROTATION_ANGLES = list(ROTATION_LABELS.keys())


# Función de preprocesamiento y rotación
def preprocess_image(file_path, image_size=224):
    # Cargar imagen y decodificar
    image = tf.io.read_file(file_path)
    image = tf.image.decode_jpeg(image, channels=3)

    # Redimensionar
    image = tf.image.resize(image, [image_size, image_size])

    # Elegir aleatoriamente una rotación
    angle = random.choice(ROTATION_ANGLES)
    label = ROTATION_LABELS[angle]

    # Rotar usando tf.image.rot90
    k = angle // 90  # 0->0, 90->1, 180->2, 270->3
    image = tf.image.rot90(image, k=k)

    # Normalizar a [-1, 1]
    image = (image / 127.5) - 1.0

    return image, label


# Crear tf.data.Dataset
def create_rotnet_dataset(image_dir, image_size=224, batch_size=32):
    image_paths = [
        os.path.join(image_dir, fname)
        for fname in os.listdir(image_dir)
        if fname.lower().endswith((".jpg", ".jpeg", ".png"))
    ]

    # Wrap Python preprocessing in tf.data.Dataset
    dataset = tf.data.Dataset.from_tensor_slices(image_paths)

    def tf_wrapper(file_path):
        image, label = tf.py_function(func=preprocess_image, inp=[file_path, image_size], Tout=[tf.float32, tf.int32])
        image.set_shape([image_size, image_size, 3])
        label.set_shape([])
        return image, label

    dataset = dataset.map(tf_wrapper, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.shuffle(buffer_size=1000)
    dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    return dataset
