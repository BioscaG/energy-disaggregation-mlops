# Resumen de Cambios - Modelo de Desagregación de Energía

## Objetivo Alcanzado ✓

Tu modelo ahora predice la potencia de un dispositivo específico (**appliance**) basándose en la potencia total del edificio (**mains power**), en lugar de solo reconstruir la entrada.

**Antes:** Input → Autoencoder → Output (copia de Input)
**Ahora:** Mains Power → Model → Appliance Power

---

## Archivos Modificados

### 1. `src/energy_dissagregation_mlops/data.py`

**Cambios:**
- `PreprocessConfig`: Reemplazado `meter=1` con `meter_mains=1` y `meter_appliance=2`
- `MyDataset.__getitem__()`: Ahora retorna `(x, y)` en lugar de solo `x`
  - `x`: Potencia total (normalized)
  - `y`: Potencia del dispositivo (normalized)
- `preprocess()`: Carga ambos metros, alinea sus índices, normaliza cada uno por separado y guarda ambas señales en los chunks `.npz`

**Impacto:**
- Los datos preprocesados ahora contienen pares entrada-salida
- Soporta cambiar qué dispositivo predecir modificando `meter_appliance`

### 2. `src/energy_dissagregation_mlops/train.py`

**Cambios:**
- Loop de entrenamiento actualizado para recibir `(x, y)` del dataloader
- Variable `y` ahora es el **target real** del dispositivo, no una copia de `x`
- MSELoss compara `y_hat` (predicción) con `y` (potencia real del dispositivo)

**Impacto:**
- El modelo ahora entrena para predecir potencia del dispositivo, no reconstruir entrada
- La función `train()` funciona como desagregación de energía (NILM)

### 3. `src/energy_dissagregation_mlops/evaluate.py`

**Cambios:**
- Loop de evaluación actualizado para usar `(x, y)` del dataloader
- Etiquetas del gráfico actualizadas:
  - "Original (Mains)" → "Appliance (Target)"
  - "Reconstructed" → "Appliance (Predicted)"
  - Título: "Signal Reconstruction Evaluation" → "Energy Disaggregation: Appliance Power Prediction"

**Impacto:**
- Las métricas (MSE, MAE, RMSE) ahora evalúan capacidad de predecir potencia del dispositivo
- Los gráficos visualizan correctamente la predicción del dispositivo

### 4. `src/energy_dissagregation_mlops/model.py`

**Cambios:**
- Docstring actualizado para aclarar que es un modelo de desagregación (NILM)

**Impacto:**
- Claridad conceptual del propósito del modelo

---

## Flujo de Datos Completo

```
RAW DATA (H5)
    ↓
1. PREPROCESS
   - Lee meter1 (mains) y meter2 (appliance)
   - Alinea índices temporales
   - Normaliza cada señal: (x - mean) / std
   - Guarda en chunks: x, y, timestamps
    ↓
2. DATASET
   - Carga ventanas de tamaño 1024
   - Retorna (x, y) pares
    ↓
3. TRAINING
   Input x (mains) → Model → Output ŷ
                              ↓
                    Compara con y (appliance)
                           ↓
                      Loss = MSE(ŷ, y)
    ↓
4. EVALUATION
   - Predice potencia del dispositivo
   - Compara con valores reales
   - Calcula: MSE, RMSE, MAE
   - Visualiza: predicción vs target
```

---

## Cómo Usar

### Preprocesar datos para un dispositivo específico:
```python
cfg = PreprocessConfig(
    meter_mains=1,        # Entrada (siempre)
    meter_appliance=2,    # Cambiar para diferentes dispositivos
)
dataset.preprocess("data/processed", cfg)
```

### Cambiar entre dispositivos:
Para predecir un dispositivo diferente (ej. meter3):
1. Cambia `meter_appliance=3` en `PreprocessConfig`
2. Ejecuta preprocesamiento nuevamente
3. Entrena un nuevo modelo

---

## Archivo de Documentación

Se creó `docs/ENERGY_DISAGGREGATION.md` con:
- Descripción arquitectónica
- Pipeline de datos detallado
- Configuración de dispositivos
- Ejemplos de uso CLI

Se creó `scripts/example_workflow.py` con:
- Ejemplo completo de flujo end-to-end
- Preprocesamiento, entrenamiento y evaluación

---

## Verificación ✓

- Sintaxis verificada en todos los archivos Python
- Todas las dependencias existentes se mantienen
- Código compatible con la infraestructura existente (CLI, tasks, etc.)
