# Resultados de calibración (Fase 2)

Línea base de la heurística geométrica (MVP), medida con `tools/evaluar.py`.
Evaluación a nivel de clip: una predicción es acierto si el detector dispara en
un clip/segmento positivo; los disparos en clips normales son falsas alarmas.

## Sets de validación

| Set | Origen | Cámara | Resolución | Clips |
|---|---|---|---|---|
| Vista única | Dataset *Shoplifting* staged | Altura de los ojos | 640×480 | 92 hurto + 90 normal |
| Multi-vista | DCSASS (UCF-Crime) | CCTV cenital (varias escenas) | 320×240 | 155 pos + 155 neg (subclips 4s) |

## Tabla precision/recall

**Vista única (640×480), `consecutive=8`, filtro de postura ON:**

| umbral | precision | recall | F1 |
|---|---|---|---|
| **0.40** | 0.561 | 0.370 | **0.446** |
| 0.50 | 0.590 | 0.337 | 0.429 |
| 0.60 | 0.596 | 0.326 | 0.422 |

**Multi-vista DCSASS (320×240), `consecutive=6`:**

| umbral | precision | recall | F1 |
|---|---|---|---|
| **0.40** | 0.433 | 0.110 | **0.175** |
| 0.50 | 0.453 | 0.103 | 0.168 |
| 0.60 | 0.425 | 0.090 | 0.149 |

## Conclusiones

- **Umbral elegido (con datos, no a ojo):** `near_score_threshold = 0.40` da el
  mejor F1 en ambos sets. En despliegue real conviene subirlo (0.50–0.60) si se
  prioriza precision sobre recall: cada señal la revisa una persona, así que
  importa no saturarla de falsas alarmas.
- **El recall se hunde de ~37% a ~11%** al pasar de cámara a la altura del torso
  a CCTV cenital. El enfoque **geométrico sobre pose es muy sensible al ángulo de
  cámara**: desde el techo, muñeca y cinturilla se proyectan juntas y la regla
  "mano delante de la cintura" deja de discriminar.
- **Implicación de despliegue:** colocar la cámara a la altura del torso /
  ligeramente elevada (no cenital) para que la heurística rinda. Con CCTV clásico
  de techo, este MVP no basta.
- **Implicación de roadmap:** estos números son la línea base contra la que medir
  el **clasificador temporal aprendido de la Fase 3** (LSTM/ST-GCN), que aprende
  el patrón de movimiento en lugar de depender de una regla geométrica fija.

## Reproducir

```bash
# Genera labels desde carpetas Shoplifting/Normal y evalúa
python tools/labels_desde_carpetas.py --root "datos/clips/<dataset>" \
    --positivo Shoplifting --salida datos/labels.csv
python tools/evaluar.py --videos-dir "datos/clips/<dataset>" \
    --labels datos/labels.csv --device 0 --sweep 0.40 0.45 0.50 0.55 0.60
```
