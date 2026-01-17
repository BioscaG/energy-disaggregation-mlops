# Guía Rápida: Tu Modelo de Desagregación de Energía

## ¿Qué cambió?

Tu modelo ahora hace **desagregación de energía (NILM)**:

```
ANTES (Autoencoder):
┌─────────────┐      ┌─────────────┐      ┌─────────────┐
│ Mains Power │ --→  │   Modelo    │  --→ │ Mains Power │
└─────────────┘      └─────────────┘      └─────────────┘
                     (Reconstruye entrada)

AHORA (Desagregación):
┌─────────────┐      ┌─────────────┐      ┌───────────────────┐
│ Mains Power │ --→  │   Modelo    │  --→ │ Appliance Power   │
└─────────────┘      └─────────────┘      │ (Dispositivo)     │
(Input)              (CNN 1D)              └───────────────────┘
                                           (Target)
```

---

## Archivos Modificados

| Archivo | Cambio |
|---------|--------|
| `data.py` | PreprocessConfig ahora tiene `meter_mains` y `meter_appliance` |
| `data.py` | Dataset retorna pares `(x, y)` en lugar de solo `x` |
| `train.py` | Loop entrena con `y` real (potencia del dispositivo) |
| `evaluate.py` | Evalúa predicción de potencia del dispositivo |
| `model.py` | Docstring actualizado |

---

## Uso Práctico

### 1. Preprocesar datos

```bash
# Para predecir dispositivo meter2
python -m energy_dissagregation_mlops.data preprocess \
  data/raw/ukdale.h5 \
  data/processed
```

El script automáticamente usa:
- `meter_mains=1` (entrada: potencia total)
- `meter_appliance=2` (salida: dispositivo a predecir)

Para cambiar dispositivo, modifica `scripts/example_workflow.py` línea 27:
```python
meter_appliance=3,  # o 4, 5, etc.
```

### 2. Entrenar

```bash
python -m energy_dissagregation_mlops.train \
  --preprocessed_folder data/processed \
  --epochs 10 \
  --lr 1e-3
```

El modelo aprenderá a predecir potencia del dispositivo.

### 3. Evaluar

```bash
python -m energy_dissagregation_mlops.evaluate \
  --preprocessed_folder data/processed \
  --checkpoint_path models/best.pt
```

Obtendrás métricas como MSE, RMSE, MAE de la predicción del dispositivo.

---

## ¿Cómo funciona el preprocesamiento?

El archivo `data.py` ahora:

1. **Lee dos metros** del dataset UK-DALE:
   - `meter_mains`: Potencia total del edificio (input)
   - `meter_appliance`: Potencia de un dispositivo (target)

2. **Alinea datos**: Si uno tiene más muestras, los ajusta

3. **Normaliza**: Cada signal se normaliza por separado
   ```
   x_normalized = (x - mean_x) / std_x
   y_normalized = (y - mean_y) / std_y
   ```

4. **Guarda ventanas**: Cada archivo `.npz` contiene:
   ```
   chunk_0000.npz:
     - x: (N, 1024)  # Mains power windows
     - y: (N, 1024)  # Appliance power windows
     - t: (N, 1024)  # Timestamps
   ```

---

## Dataset Returns

```python
x, y = dataset[0]
# x.shape = [1, 256]  # 1 channel (mains), 256 timesteps
# y.shape = [1, 256]  # 1 channel (appliance), 256 timesteps
```

En entrenamiento (batch):
```python
for x, y in train_loader:
    # x: [B, 1, T]  # Mains (B=batch size, T=timesteps)
    # y: [B, 1, T]  # Appliance (target)
    y_hat = model(x)
    loss = MSELoss(y_hat, y)
```

---

## Cambiar Dispositivo

Para predecir un dispositivo diferente:

### Opción 1: Línea de comandos
```python
# En scripts/example_workflow.py, línea 27:
meter_appliance=3,  # Cambiar a 3, 4, 5, etc.
```

### Opción 2: Crear un script personalizado
```python
from energy_dissagregation_mlops.data import MyDataset, PreprocessConfig
from pathlib import Path

cfg = PreprocessConfig(
    meter_mains=1,
    meter_appliance=5,  # Tu dispositivo deseado
    window_size=1024,
)

dataset = MyDataset(Path("data/raw/ukdale.h5"))
dataset.preprocess(Path("data/processed_meter5"), cfg=cfg)
```

Luego entrena:
```bash
python -m energy_dissagregation_mlops.train \
  --preprocessed_folder data/processed_meter5
```

---

## Verificación

Para verificar que todo funciona:

```bash
python scripts/example_workflow.py
```

Esto ejecutará:
1. ✓ Preprocesamiento
2. ✓ Entrenamiento (5 epochs)
3. ✓ Evaluación con gráficos

---

## Preguntas Frecuentes

**P: ¿El modelo ahora es mejor?**
R: Depende del dataset. Ahora entrena en la tarea correcta (desagregación) en lugar de reconstruir la entrada.

**P: ¿Puedo predecir múltiples dispositivos a la vez?**
R: No con esta arquitectura (output es single appliance). Para múltiples dispositivos necesitarías cambiar el modelo para tener múltiples outputs.

**P: ¿Cómo desnormalizo las predicciones?**
R: Las estadísticas se guardan en `meta.npz`. Para desnormalizar:
```python
meta = np.load("data/processed/meta.npz", allow_pickle=True)
y_hat_normalized = model(x)
y_hat_real = y_hat_normalized * meta["std_appliance"] + meta["mean_appliance"]
```

**P: ¿Qué es `stride` en la configuración?**
R: Es el paso entre ventanas. `stride=256` con `window_size=1024` significa ventanas solapadas al 75%.

---

## Documentación Completa

Para más detalles, ver:
- `docs/ENERGY_DISAGGREGATION.md`: Arquitectura y pipeline detallado
- `CAMBIOS_REALIZADOS.md`: Listado completo de cambios
- `scripts/example_workflow.py`: Ejemplo end-to-end
