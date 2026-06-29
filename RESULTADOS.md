# Resultados de calibración (Fase 2)

Línea base de la heurística geométrica (MVP), medida con `tools/evaluar.py`.
Evaluación a nivel de clip: una predicción es acierto si el detector dispara en
un clip/segmento positivo; los disparos en clips normales son falsas alarmas.

## Sets de validación

| Set | Origen | Cámara | Resolución | Clips |
|---|---|---|---|---|
| Vista única | Dataset *Shoplifting* staged | Altura de los ojos | 640×480 | 92 hurto + 90 normal |
| CCTV chaqueta | CCTV_Shoplifting (sintético) | Elevada ~45° | 544×544 | 4 hurto + 4 normal |
| Multi-vista | DCSASS (UCF-Crime) | CCTV cenital (varias escenas) | 320×240 | 155 pos + 155 neg (subclips 4s) |

## Mejora de cobertura de gesto

La línea base inicial solo detectaba la ocultación BAJA (mano a la cinturilla).
Al medir contra datos reales se vio que se perdía la ocultación ALTA (meter algo
en la chaqueta/pecho). Se añadió una segunda ancla en el torso
(`chest_anchor_ratio`), cubriendo toda la franja frontal del cuerpo.

**Set vista única (640×480), antes vs. después del arreglo:**

| versión | umbral | precision | recall | F1 |
|---|---|---|---|---|
| solo cinturilla | 0.40 | 0.561 | 0.370 | 0.446 |
| + pecho (cinturilla∪pecho) | 0.40 | 0.655 | 0.826 | 0.731 |
| + pecho — **mejor punto** | **0.50** | **0.712** | **0.815** | **0.760** |

El recall pasa de 0.37 a 0.83 (de 34/92 a 75–76/92 hurtos detectados) y la
precision sube de 0.56 a 0.71. El umbral óptimo se mueve a 0.50.

## Tabla precision/recall (heurística mejorada)

**Vista única (640×480), `consecutive=8`, filtro de postura ON:**

| umbral | precision | recall | F1 |
|---|---|---|---|
| 0.40 | 0.655 | 0.826 | 0.731 |
| **0.50** | **0.712** | 0.815 | **0.760** |
| 0.60 | 0.693 | 0.674 | 0.683 |

**CCTV chaqueta (544×544, ángulo elevado), n=8 (cualitativo):**

| umbral | precision | recall | F1 |
|---|---|---|---|
| 0.40 | 0.692 | 1.000 | 0.818 |
| 0.50 | 0.833 | 0.500 | 0.625 |

**Multi-vista DCSASS (320×240, CCTV cenital), `consecutive=6`:**

| umbral | precision | recall | F1 |
|---|---|---|---|
| 0.40 | 0.433 | 0.110 | 0.175 |
| 0.50 | 0.453 | 0.103 | 0.168 |

## Conclusiones

- **Umbral elegido (con datos, no a ojo):** `near_score_threshold = 0.50` da el
  mejor F1 en cámara a la altura del torso (vista única). En despliegue se puede
  subir si se prioriza precision: cada señal la revisa una persona.
- **Dos límites distintos de la heurística, identificados midiendo:**
  1. *Cobertura de gesto* — se resolvió añadiendo el ancla de pecho (recall
     ×2 en vista única; 4/4 en el set de chaqueta).
  2. *Ángulo de cámara* — el recall se hunde a ~0.11 en CCTV cenital (320×240):
     desde el techo, muñeca y torso se proyectan juntos y la geometría deja de
     discriminar. Esto **no** se arregla con más reglas: es la motivación de la
     Fase 3.
- **Implicación de despliegue:** colocar la cámara a la altura del torso /
  ligeramente elevada (no cenital). Con CCTV clásico de techo, el MVP no basta.
- **Implicación de roadmap:** estos números son la línea base contra la que medir
  el clasificador temporal aprendido de la **Fase 3** (LSTM/ST-GCN), que aprende
  el patrón de movimiento en lugar de depender de reglas geométricas fijas.

## Fase 3 — clasificador temporal aprendido (PoseLSTM)

Se entrenó un LSTM sobre secuencias de pose normalizada (ventanas de 32 frames,
51 features/frame) con los 3 datasets, división **por vídeo** (sin fuga de datos)
y pesos de clase por el desbalance. Inferencia en streaming vía
`LearnedConcealmentDetector` (misma interfaz que la heurística, intercambiable en
el pipeline).

**Comparación clip-level sobre los MISMOS 40 vídeos held-out del set 640
(19 hurto + 21 normal):**

| Detector | Precision | Recall | F1 | mejor umbral |
|---|---|---|---|---|
| Heurística (cinturilla∪pecho) | 0.757 | 0.895 | 0.820 | 0.50 |
| **PoseLSTM aprendido** | **0.837** | **0.947** | **0.889** | 0.80 (prob) |

El clasificador **supera a la heurística en las tres métricas** y mantiene recall
0.947 (18/19) en todo el rango de umbrales: subir el umbral limpia falsas
alarmas sin perder recall. Cumple el criterio de "hecho" de la Fase 3
(el modelo aprendido bate la línea base heurística, documentado).

Entrenamiento (validación a nivel de ventana): mejor F1 = 0.703.

```bash
# Preparar datasets, entrenar y comparar
python tools/preparar_dataset.py --videos-dir "<dataset>" --labels <labels.csv> \
    --out datos/ds_X.npz --device 0
python tools/entrenar.py --data datos/ds_*.npz --out modelos/poselstm.pt --device 0
python tools/evaluar.py --videos-dir "<dataset>" --labels <labels.csv> \
    --only datos/val_ds_640.txt --learned modelos/poselstm.pt --device 0 \
    --sweep 0.4 0.5 0.6 0.7 0.8
```

## Reproducir

```bash
python tools/labels_desde_carpetas.py --root "datos/clips/<dataset>" \
    --positivo Shoplifting --salida datos/labels.csv
python tools/evaluar.py --videos-dir "datos/clips/<dataset>" \
    --labels datos/labels.csv --device 0 --sweep 0.40 0.45 0.50 0.55 0.60
```
