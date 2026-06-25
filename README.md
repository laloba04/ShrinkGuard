# ShrinkGuard

Detección de **gestos de ocultación** (mano a cinturilla / bolsillo / bolso) en
vídeo, mediante estimación de pose multi-persona con seguimiento. Pensado como
sistema de *prevención de pérdidas* (retail shrinkage) con un principio
irrenunciable: el sistema genera **señales de sospecha para revisión humana**,
nunca decisiones automáticas sobre personas.

> ⚠️ Esto es una **señal**, no una acusación. Toda alerta la valida una persona.
> Antes de cualquier despliegue real, leer la sección de [Ética y legalidad](#ética-y-legalidad).

## Qué hace (Fase 1)

- Lee un vídeo (archivo o webcam).
- Estima pose de todas las personas con `YOLO11-pose` y las sigue con `ByteTrack`
  (un `track_id` estable por persona).
- Aplica una heurística temporal: si la muñeca de alguien permanece N frames
  delante de la cinturilla, dispara una señal.
- Anota el vídeo y guarda un recorte con marca de tiempo en `salidas/revision/`
  más un `eventos.csv`, para que un humano lo revise.

La heurística es deliberadamente sustituible por un clasificador aprendido
(LSTM / ST-GCN) en la Fase 3 sin tocar el resto del pipeline.

## Arquitectura

```mermaid
flowchart TD
    A["Stream de vídeo<br/>archivo o RTSP"] --> B["Detección + tracking<br/>YOLO11-pose + ByteTrack"]
    B --> C["Pose multi-persona<br/>17 keypoints por track_id"]
    C --> D["Buffer temporal<br/>estado por persona"]
    D --> E["Heurística de ocultación<br/>(Fase 3: LSTM / ST-GCN)"]
    E --> F["Zona + alerta<br/>recorte + CSV"]
    F --> G["Revisión humana<br/>decisión final"]
```

El acoplamiento con Ultralytics vive solo en `src/pose.py`. El detector
(`src/concealment.py`) trabaja únicamente con `numpy`, por lo que se testea sin
modelos ni vídeo.

```
shrinkguard/
├── main.py                 # CLI
├── src/
│   ├── config.py           # parámetros del pipeline
│   ├── pose.py             # YOLO11-pose + tracking  (única dependencia de Ultralytics)
│   ├── concealment.py      # heurística + estado temporal  (solo numpy, testeable)
│   ├── visualizer.py       # overlays con OpenCV
│   └── pipeline.py         # bucle principal
└── tests/
    └── test_concealment.py # 6 tests con keypoints sintéticos
```

## Instalación

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

La primera ejecución descarga sola el modelo `yolo11n-pose.pt`.

## Uso

```bash
# Vídeo grabado
python main.py --source data/tienda.mp4

# Webcam (Mac con Apple Silicon)
python main.py --source 0 --device mps

# Sin ventana, guardando vídeo anotado (p. ej. en servidor)
python main.py --source data/tienda.mp4 --save salidas/anotado.mp4 --no-window
```

Las evidencias para revisión quedan en `salidas/revision/` (recortes + `eventos.csv`).

## Tests

```bash
python -m pytest -q          # o:  python tests/test_concealment.py
```

## Ética y legalidad

Un sistema que infiere intención de hurto a partir de imágenes de personas es
sensible por diseño. Para un despliegue real en la UE / España hay que tener en
cuenta, como mínimo:

- **RGPD + LOPDGDD**: tratamiento de imágenes de personas. Requiere base legal,
  información (cartelería), minimización y plazos de conservación. Las imágenes
  de la carpeta de revisión son datos personales: cifrado, acceso restringido,
  borrado programado.
- **Guías de la AEPD sobre videovigilancia**: limitan finalidad, ubicación de
  cámaras y conservación.
- **Reglamento de IA (AI Act)**: parte de estos usos pueden considerarse de alto
  riesgo; conviene documentar el sistema, sus límites y su evaluación.
- **Decisión humana obligatoria**: el sistema no acusa, no identifica y no actúa.
  Solo señala para que una persona revise (human-in-the-loop).
- **Falsos positivos**: rascarse, colocarse la ropa o guardar el móvil se parecen
  a "ocultar". Por eso se mide *precision/recall* y se ajustan umbrales; nunca se
  toma la salida como verdad.

> Este repo es un proyecto de aprendizaje/portfolio. No constituye asesoría legal.

## Hoja de ruta

Ver [`GUIA.md`](GUIA.md) para el plan por fases (de la heurística actual al
clasificador temporal, el panel de revisión y el despliegue en tiempo real).

## Licencia

MIT (pendiente de añadir `LICENSE`). Ultralytics YOLO se distribuye bajo AGPL-3.0:
revisa sus términos si piensas usarlo en producción.
