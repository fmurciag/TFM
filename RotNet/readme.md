# 🌱 RotNet - Self-Supervised Learning for Plant Classification

Este proyecto implementa **RotNet**, un enfoque de aprendizaje auto-supervisado (SSL) que entrena una red neuronal para predecir la **orientación rotada** de una imagen. Tras el preentrenamiento, se reutiliza el encoder para una tarea de **clasificación supervisada real** (p. ej., 30 clases de tipos de plantas).

---

## 📌 Objetivo

- Aprender representaciones útiles sin etiquetas utilizando una tarea pretexto simple: **predecir la rotación** de la imagen (0°, 90°, 180°, 270°).
- Usar el encoder preentrenado para mejorar la clasificación con menos datos etiquetados.

---

## 🧩 Fase 1 - Preentrenamiento (Tarea Pretexto)

### 1. Preparación de datos

- Para cada imagen original, generar 4 versiones:
  - 0° → etiqueta: 0
  - 90° → etiqueta: 1
  - 180° → etiqueta: 2
  - 270° → etiqueta: 3
- Almacenar (imagen_rotada, etiqueta_rotación) como nuevos pares de entrenamiento.

### 2. Arquitectura

- Red CNN (por ejemplo, ResNet18 o una CNN simple).
- Modificar la **última capa fully connected** para que tenga 4 neuronas (softmax) → una por cada rotación.


### 3. Entrenamiento

- **Función de pérdida**: `CrossEntropyLoss`.
- **Optimización**: `Adam` o `SGD`.
- **Objetivo**: minimizar la pérdida de clasificación de rotación.

---

## 🎯 Fase 2 - Clasificación Supervisada (Downstream Task)

### 1. Cargar encoder preentrenado

- Usar la red entrenada en la Fase 1.
- **Eliminar la última capa de clasificación de rotación (FC(4))**.
- **Congelar o ajustar el encoder** según el experimento.

### 2. Añadir nueva capa de clasificación

- Insertar una nueva capa FC con salida del número de clases reales (p. ej., 30 para 30 tipos de plantas).
- Entrenar esta nueva parte con imágenes etiquetadas.


### 3. Métricas

- Accuracy, precisión, recall, F1-score, matriz de confusión.
- Comparar contra un modelo entrenado desde cero (baseline supervisado puro).


---
## 📦 Requisitos

- Python ≥ 3.8
- PyTorch / TensorFlow
- Albumentations / torchvision (para augmentaciones y rotaciones)

---

## 💡 Ventajas de RotNet

- No requiere etiquetas humanas para preentrenar.
- Fácil de implementar y entender.
- Compatible con cualquier CNN.

---

## 🔁 Flujo de trabajo general

1. Preentrenamiento SSL con imágenes rotadas (sin etiquetas reales).
2. Guardar pesos del encoder.
3. Cargar encoder, reemplazar capa de salida.
4. Entrenar con pocas imágenes etiquetadas reales.
5. Comparar con entrenamiento supervisado desde cero.

---

## 📚 Referencias

- Gidaris, Spyros, et al. "Unsupervised Representation Learning by Predicting Image Rotations." *ICLR 2018*.  
- Arash Khoeini. [Medium: Self-Supervised Learning](https://arashk.medium.com)


## 🧪 Preprocesamiento - Generación de Tarea Pretexto para RotNet

El objetivo del preprocesamiento es convertir un conjunto de imágenes **sin etiquetas** en un dataset válido para la tarea auto-supervisada de **clasificación de rotaciones**.

### 🔁 Pasos del preprocesamiento:

1. **Carga de imágenes originales**
   - Leer todas las imágenes del dataset base (carpeta `data/raw/`).
   - Aplicar transformaciones básicas: redimensionar (por ejemplo a 224x224), normalizar, etc.

2. **Generar versiones rotadas**
   - Para cada imagen original, generar 4 copias:
     - Rotación 0°  → etiqueta 0
     - Rotación 90° → etiqueta 1
     - Rotación 180° → etiqueta 2
     - Rotación 270° → etiqueta 3

3. **Guardar o preparar en memoria**
   - Puedes guardar las imágenes rotadas en disco (carpeta `data/rotated/`) o crearlas **on-the-fly** durante el entrenamiento mediante un `Dataset` personalizado (más eficiente y flexible).

4. **Estructura de datos resultante**
   - Cada elemento del dataset se representa como:
     ```
     {
       "image": imagen_rotada (tensor o array),
       "label": clase_rotación (0, 1, 2, 3)
     }
     ```

### ✨ Ejemplo conceptual en pseudocódigo:

```python
for img_path in raw_images:
    image = load_image(img_path)
    for angle, label in zip([0, 90, 180, 270], [0, 1, 2, 3]):
        rotated = rotate_image(image, angle)
        dataset.append((rotated, label))
