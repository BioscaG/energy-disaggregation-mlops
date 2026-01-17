# Energy Disaggregation Model (NILM)

## Overview

El modelo ahora está configurado para realizar **desagregación de energía** (NILM - Non-Intrusive Load Monitoring). El modelo predice la potencia de un dispositivo específico a partir de la potencia total (mains).

## Architecture

- **Entrada (Input)**: Potencia total del edificio (mains power) - meter1
- **Salida (Output)**: Potencia del dispositivo específico (appliance power) - meter2+
- **Modelo**: CNN 1D con 4 capas de convolución

## Data Pipeline

### 1. Preprocesamiento (`data.py`)

El módulo `PreprocessConfig` ahora carga dos metros en lugar de uno:

```python
cfg = PreprocessConfig(
    building=1,
    meter_mains=1,         # Potencia total
    meter_appliance=2,     # Dispositivo a predecir (meter2, meter3, etc.)
    window_size=1024,
    stride=256,
    normalize=True,        # Z-score normalization
)
```

**Cambios principales:**
- Lee tanto `meter_mains` como `meter_appliance` del dataset
- Alinea los índices temporales de ambos metros
- Normaliza ambas señales usando estadísticas globales
- Guarda `x` (mains) e `y` (appliance) en cada ventana

### 2. Dataset (`MyDataset`)

El dataset ahora retorna pares (entrada, salida):

```python
x, y = dataset[i]
# x: [1, T] - potencia total (normalized)
# y: [1, T] - potencia del dispositivo (normalized)
```

### 3. Entrenamiento (`train.py`)

El loop de entrenamiento ahora recibe tanto entrada como salida:

```python
for x, y in train_loader:
    # x: [B, 1, T] - mains power
    # y: [B, 1, T] - appliance power (target)
    y_hat = model(x)
    loss = MSELoss(y_hat, y)
```

### 4. Evaluación (`evaluate.py`)

La evaluación compara las predicciones del dispositivo con los valores reales:

```python
for x, y in test_loader:
    y_hat = model(x)  # Predice potencia del dispositivo
    mse = MSELoss(y_hat, y)  # Compara con potencia real del dispositivo
```

## Archivos Modificados

- `src/energy_dissagregation_mlops/data.py`: Soporte para dos metros
- `src/energy_dissagregation_mlops/train.py`: Entrenamiento con entrada-salida
- `src/energy_dissagregation_mlops/evaluate.py`: Evaluación para desagregación
- `src/energy_dissagregation_mlops/model.py`: Docstring actualizado

## Uso

### Preprocesar datos

```bash
python -m energy_dissagregation_mlops.data preprocess \
  data/raw/ukdale.h5 \
  data/processed
```

### Entrenar modelo

```bash
python -m energy_dissagregation_mlops.train \
  --preprocessed_folder data/processed \
  --epochs 10 \
  --batch_size 32
```

### Evaluar modelo

```bash
python -m energy_dissagregation_mlops.evaluate \
  --preprocessed_folder data/processed \
  --checkpoint_path models/best.pt
```

## Métricas

- **MSE**: Error cuadrático medio (normalizado)
- **RMSE**: Raíz del error cuadrático medio
- **MAE**: Error absoluto medio

## Configuración de Dispositivos

Para cambiar qué dispositivo predecir, modifica `meter_appliance` en `PreprocessConfig`:

- `meter_appliance=2`: Dispositivo 2
- `meter_appliance=3`: Dispositivo 3
- etc.

Luego vuelve a ejecutar el preprocesamiento para generar nuevos chunks con el dispositivo deseado.

## Notas

- Los datos se normalizan usando z-score (media 0, std 1)
- Las estadísticas de normalización se guardan en `meta.npz`
- El modelo puede usarse para diferentes dispositivos simplemente reentrenando con diferentes `meter_appliance`
