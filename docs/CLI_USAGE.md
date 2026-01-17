# CLI Usage Examples - Energy Disaggregation Model

## Overview

El modelo ahora incluye CLI completamente funcional con soporte para desagregación de energía.

## Available Commands

### 1. `preprocess` - Prepare data

**Preprocesa datos desde UK-DALE HDF5 para entrenamiento**

```bash
python -m energy_dissagregation_mlops.cli preprocess \
  --data-path data/raw/ukdale.h5 \
  --output-folder data/processed
```

**Con parámetros personalizados:**

```bash
python -m energy_dissagregation_mlops.cli preprocess \
  --data-path data/raw/ukdale.h5 \
  --output-folder data/processed_meter3 \
  --building 1 \
  --meter-mains 1 \
  --meter-appliance 3 \
  --window-size 1024 \
  --stride 256 \
  --resample-rule 6S \
  --normalize
```

**Parámetros:**
- `--data-path` (required): Ruta al archivo ukdale.h5
- `--output-folder` (required): Carpeta de salida para chunks
- `--building`: Número de edificio UK-DALE (default: 1)
- `--meter-mains`: Meter para potencia total (default: 1) ⭐ **NUEVO**
- `--meter-appliance`: Meter para dispositivo a predecir (default: 2) ⭐ **NUEVO**
- `--window-size`: Tamaño de ventana en muestras (default: 1024)
- `--stride`: Paso entre ventanas (default: 256)
- `--resample-rule`: Regla de resampleo Pandas (default: "6S")
- `--power-type`: Tipo de potencia: apparent/active (default: "apparent")
- `--normalize/--no-normalize`: Normalizar z-score (default: true)

**Ejemplos de uso común:**

```bash
# Predecir meter 2 (frigorífico, típico)
python -m energy_dissagregation_mlops.cli preprocess \
  --data-path data/raw/ukdale.h5 \
  --output-folder data/processed_fridge \
  --meter-appliance 2

# Predecir meter 5 (otro dispositivo)
python -m energy_dissagregation_mlops.cli preprocess \
  --data-path data/raw/ukdale.h5 \
  --output-folder data/processed_meter5 \
  --meter-appliance 5

# Sin resampleo (mantener frecuencia original)
python -m energy_dissagregation_mlops.cli preprocess \
  --data-path data/raw/ukdale.h5 \
  --output-folder data/processed_original \
  --resample-rule none
```

---

### 2. `train` - Train model

**Entrena el modelo con datos preprocesados**

```bash
python -m energy_dissagregation_mlops.cli train \
  --preprocessed-folder data/processed
```

**Con parámetros personalizados:**

```bash
python -m energy_dissagregation_mlops.cli train \
  --preprocessed-folder data/processed \
  --epochs 20 \
  --batch-size 64 \
  --lr 0.001 \
  --num-workers 4 \
  --device cuda
```

**Parámetros:**
- `--preprocessed-folder`: Carpeta con datos preprocesados (default: data/processed)
- `--epochs`: Número de épocas (default: 5)
- `--batch-size`: Tamaño de batch (default: 32)
- `--lr`: Learning rate (default: 0.001)
- `--num-workers`: Número de workers del dataloader (default: 2)
- `--device`: Device (auto/cpu/cuda, default: auto)

**Ejemplos:**

```bash
# Entrenamiento rápido (CPU)
python -m energy_dissagregation_mlops.cli train \
  --epochs 5 \
  --device cpu

# Entrenamiento completo (GPU)
python -m energy_dissagregation_mlops.cli train \
  --preprocessed-folder data/processed \
  --epochs 50 \
  --batch-size 128 \
  --lr 5e-4 \
  --device cuda

# Ajuste fino (learning rate bajo)
python -m energy_dissagregation_mlops.cli train \
  --epochs 10 \
  --lr 1e-4 \
  --batch-size 16
```

---

### 3. `evaluate` - Evaluate model

**Evalúa el modelo entrenado**

```bash
python -m energy_dissagregation_mlops.cli evaluate \
  --preprocessed-folder data/processed \
  --checkpoint-path models/best.pt \
  --plot-results
```

**Parámetros:**
- `--preprocessed-folder`: Carpeta con datos preprocesados (default: data/processed)
- `--checkpoint-path`: Ruta al checkpoint del modelo (default: models/best.pt)
- `--batch-size`: Tamaño de batch (default: 32)
- `--device`: Device (auto/cpu/cuda, default: auto)
- `--plot-results/--no-plot-results`: Guardar gráfico (default: false)

**Ejemplos:**

```bash
# Evaluación simple
python -m energy_dissagregation_mlops.cli evaluate

# Con visualización
python -m energy_dissagregation_mlops.cli evaluate \
  --plot-results

# GPU
python -m energy_dissagregation_mlops.cli evaluate \
  --device cuda \
  --plot-results

# Custom checkpoint
python -m energy_dissagregation_mlops.cli evaluate \
  --checkpoint-path models/checkpoint_epoch10.pt \
  --preprocessed-folder data/processed_meter5 \
  --plot-results
```

---

### 4. `download` - Download UK-DALE dataset

**Descarga el dataset UK-DALE**

```bash
python -m energy_dissagregation_mlops.cli download \
  --target-dir data/raw
```

---

## Complete Workflows

### Workflow 1: Basic (All in one)

```bash
# 1. Preprocesar
python -m energy_dissagregation_mlops.cli preprocess \
  --data-path data/raw/ukdale.h5 \
  --output-folder data/processed

# 2. Entrenar
python -m energy_dissagregation_mlops.cli train \
  --epochs 10

# 3. Evaluar
python -m energy_dissagregation_mlops.cli evaluate \
  --plot-results
```

### Workflow 2: Predecir dispositivo diferente

```bash
# 1. Preprocesar para meter 3
python -m energy_dissagregation_mlops.cli preprocess \
  --data-path data/raw/ukdale.h5 \
  --output-folder data/processed_meter3 \
  --meter-appliance 3

# 2. Entrenar
python -m energy_dissagregation_mlops.cli train \
  --preprocessed-folder data/processed_meter3 \
  --epochs 15

# 3. Evaluar
python -m energy_dissagregation_mlops.cli evaluate \
  --preprocessed-folder data/processed_meter3 \
  --checkpoint-path models/best.pt \
  --plot-results
```

### Workflow 3: Entrenamiento avanzado

```bash
# Preproceso con custom window size
python -m energy_dissagregation_mlops.cli preprocess \
  --data-path data/raw/ukdale.h5 \
  --output-folder data/processed_custom \
  --window-size 512 \
  --stride 128

# Entrenamiento largo con GPU
python -m energy_dissagregation_mlops.cli train \
  --preprocessed-folder data/processed_custom \
  --epochs 100 \
  --batch-size 256 \
  --lr 5e-4 \
  --device cuda \
  --num-workers 8

# Evaluación detallada
python -m energy_dissagregation_mlops.cli evaluate \
  --preprocessed-folder data/processed_custom \
  --batch-size 256 \
  --device cuda \
  --plot-results
```

---

## CLI Verification

Verificar que CLI funciona correctamente:

```bash
# Ver todas las opciones disponibles
python -m energy_dissagregation_mlops.cli preprocess --help
python -m energy_dissagregation_mlops.cli train --help
python -m energy_dissagregation_mlops.cli evaluate --help
python -m energy_dissagregation_mlops.cli download --help

# Quick test (ver que no hay errores de sintaxis)
python -c "from energy_dissagregation_mlops.cli import app; print('✓ CLI loaded successfully')"
```

---

## Key Changes to CLI

| Parámetro | Antes | Después |
|-----------|-------|---------|
| `--meter` | ✓ Un solo metro | ❌ Removido |
| `--meter-mains` | ❌ N/A | ✓ Nuevo (default: 1) |
| `--meter-appliance` | ❌ N/A | ✓ Nuevo (default: 2) |
| Funcionalidad | Reconstruir entrada | Desagregación (entrada → salida) |

---

## Tips & Tricks

### Cambiar dispositivo rápidamente

```bash
# Script bash para entrenar múltiples dispositivos
for meter in 2 3 4 5; do
  echo "Training for meter $meter..."
  python -m energy_dissagregation_mlops.cli preprocess \
    --data-path data/raw/ukdale.h5 \
    --output-folder data/processed_meter$meter \
    --meter-appliance $meter

  python -m energy_dissagregation_mlops.cli train \
    --preprocessed-folder data/processed_meter$meter \
    --epochs 10
done
```

### Monitorear memoria (GPU)

```bash
# En otra terminal, monitorear GPU
watch -n 0.1 nvidia-smi
```

### Comparar modelos

```bash
# Entrenar con diferentes hyperparameters
for lr in 1e-3 5e-4 1e-4; do
  echo "Training with lr=$lr..."
  python -m energy_dissagregation_mlops.cli train \
    --lr $lr \
    --epochs 5
done

# Evaluar todos
python -m energy_dissagregation_mlops.cli evaluate --plot-results
```
