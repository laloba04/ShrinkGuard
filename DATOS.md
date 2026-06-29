# Datos para calibrar (Fase 2)

Para medir *precision/recall* del detector hace falta un pequeño conjunto de
clips **etiquetados a mano**: unos cuantos con gesto de ocultación (positivos) y
unos cuantos de comportamiento normal (negativos). No hacen falta muchos: 10–20
clips representativos ya dan una línea base útil.

## ⚠️ Antes de descargar nada

Vídeos de personas (y más aún, de personas presuntamente hurtando) son **datos
personales sensibles**. Para un proyecto de aprendizaje:

- Usa **datasets de investigación publicados para este fin** (abajo). Evita
  scrapear cámaras de seguridad sueltas de YouTube/redes: privacidad y derechos.
- Para tu tienda real: metraje propio **con cartelería y base legal** (RGPD/AEPD),
  acceso restringido y borrado programado. Ver la sección de ética del README.

## De dónde sacar clips

- **UCF-Crime** — dataset de referencia para detección de anomalías en vídeo.
  Tiene la categoría **`Shoplifting`** (hurto en tienda), justo el caso de uso.
  Es grande; descarga solo unos cuantos clips de esa categoría.
  Buscar: *"UCF-Crime dataset Shoplifting"* (suele estar en Dropbox/Kaggle).
- **DCSASS Dataset** — clips cortos por categoría, también con `Shoplifting`.
  Más ligero que UCF-Crime para empezar.

Coloca los clips elegidos en `datos/clips/` (esta carpeta está en `.gitignore`:
no se sube metraje al repo).

## Cómo etiquetar

Mira cada clip y anota los **intervalos** (en segundos) donde ocurre el gesto de
ocultación. Vuélcalos en `datos/labels.csv` con esta cabecera:

```csv
video,inicio,fin
robo1.mp4,3.5,6.0
robo1.mp4,12.0,14.5
robo2.mp4,1.0,2.8
```

- Una fila por intervalo positivo. Un mismo vídeo puede tener varias filas.
- Los clips de `datos/clips/` que **no** aparezcan en el CSV se tratan como
  100% negativos (solo pueden generar falsas alarmas).

## Cómo evaluar y elegir umbral

```bash
python tools/evaluar.py --videos-dir datos/clips --labels datos/labels.csv \
    --sweep 0.30 0.40 0.45 0.50 0.55 0.60 --device 0
```

Imprime una tabla precision/recall/F1 por umbral y recomienda el de mejor F1.
La pose se calcula una sola vez por vídeo; el barrido de umbral es rápido.

Criterio típico en prevención de pérdidas: prioriza **precision** (pocas falsas
alarmas, porque cada señal la revisa una persona), aceptando algo menos de
recall. Elige el umbral mirando la tabla, no a ojo — ese es el "criterio de
hecho" de la Fase 2.
