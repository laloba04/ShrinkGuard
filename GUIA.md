# Guía de fases — ShrinkGuard

Plan de desarrollo de un sistema de detección de gestos de ocultación por pose,
orientado a algo cercano a producción. La filosofía es la misma que en un buen
proyecto de portfolio: **elegir una pieza acotada y profundizar**, dejando por
escrito las decisiones técnicas y, sobre todo, las consideraciones éticas y
legales que son las que distinguen un sistema serio de una demo.

Cada fase indica: **objetivo**, **entregables**, **decisiones técnicas** y
**criterio de "hecho"** (cómo sabes que has terminado).

---

## Fase 0 — Encuadre, ética y legalidad

Va primero a propósito. Si esto no está claro, lo demás no debería construirse.

**Objetivo.** Definir qué señala el sistema, qué NO hace, y bajo qué marco.

**Decisiones clave.**
- El sistema produce **señales de sospecha para revisión humana**. No identifica,
  no acusa, no actúa. La decisión es siempre de una persona.
- Datos personales por diseño: imágenes de personas. Aplica RGPD + LOPDGDD, guías
  de la AEPD sobre videovigilancia y, potencialmente, el Reglamento de IA.
- Minimización: guardar lo justo, cifrado, con borrado programado y acceso
  restringido. Nada de almacenar vídeo indefinidamente "por si acaso".

**Entregables.**
- Sección de ética/legalidad en el README (hecho).
- Una "DPIA simulada" breve (evaluación de impacto): finalidad, riesgos, medidas.
  Aunque sea un ejercicio de portfolio, demuestra criterio.

**Criterio de "hecho".** Puedes explicar en una entrevista por qué el sistema
nunca decide solo y qué obligaciones legales tendría en producción.

---

## Fase 1 — MVP heurístico  ✅ (incluido en este repo)

**Objetivo.** Tener algo que funcione de punta a punta sobre vídeo grabado.

**Qué entrega el código actual.**
- Lectura de vídeo (archivo o webcam).
- Pose multi-persona con `YOLO11-pose` + tracking con `ByteTrack` (IDs estables).
- Heurística de ocultación normalizada por el tamaño del torso (funciona a
  distintas distancias de la cámara): mano delante de la cinturilla durante N
  frames → señal.
- Estado temporal por persona con *cooldown* para no spamear alertas.
- Anotación del vídeo + guardado de recortes y `eventos.csv` para revisión humana.
- 6 tests del detector con keypoints sintéticos (sin dependencias pesadas).

**Decisiones técnicas.**
- La heurística vive en `concealment.py` y solo usa `numpy`: es testeable y
  reemplazable por un modelo aprendido sin tocar el pipeline.
- La dependencia de Ultralytics está aislada en `pose.py`.

**Criterio de "hecho".** Pasas un vídeo de prueba y se generan recortes
coherentes cuando alguien se lleva la mano a la cinturilla; los tests están en
verde.

**Limitaciones honestas.** Una heurística geométrica confunde gestos cotidianos
(ajustarse el cinturón, guardar el móvil). Es el punto de partida, no el final.

---

## Fase 2 — Robustez y configurabilidad

**Objetivo.** Reducir falsos positivos y hacer el sistema parametrizable.

**Tareas.**
- Suavizado temporal de keypoints (p. ej. media móvil / filtro) para reducir
  el ruido del estimador de pose.
- Manejo de oclusiones: qué hacer cuando faltan keypoints fiables.
- Calibración de umbrales con vídeos reales etiquetados a mano (pocos, pero
  representativos): mide *precision* y *recall* del MVP como línea base.
- Zonas de interés configurables (ROI): solo analizar ciertas áreas del frame.

**Criterio de "hecho".** Tienes una tabla con precision/recall del MVP sobre tu
set de validación y umbrales razonados, no elegidos a ojo.

---

## Fase 3 — Clasificador temporal aprendido (la chicha técnica)

**Objetivo.** Sustituir la heurística por un modelo que aprende el gesto a
partir de **secuencias** de poses.

**Enfoques.**
- **LSTM/GRU** sobre secuencias de keypoints normalizados (entrada sencilla,
  buen punto de partida).
- **ST-GCN** (Spatio-Temporal Graph Convolutional Network): trata el esqueleto
  como un grafo y modela el movimiento; es el enfoque "de referencia" para
  reconocimiento de acciones basado en esqueleto.

**Datos.**
- Datasets públicos con clase de hurto: **UCF-Crime** (tiene "Shoplifting"),
  **DCSASS**. Son ruidosos: documenta cómo los limpias y etiquetas.
- Probablemente necesites grabar/etiquetar tus propios clips (con
  consentimiento) para gestos concretos. Cuida el sesgo del dataset.

**Decisiones técnicas.**
- Trabaja siempre sobre **keypoints normalizados**, no sobre píxeles crudos:
  generaliza mejor y reduce sesgos por apariencia/ropa.
- Define bien la ventana temporal (cuántos frames forman una "acción").

**Criterio de "hecho".** El clasificador supera a la heurística de la Fase 1 en
tu set de validación, con la comparación documentada (precision/recall, matriz
de confusión).

---

## Fase 4 — Lógica de negocio y panel de revisión

**Objetivo.** Convertir señales en un flujo de revisión usable.

**Tareas.**
- Lógica de zona: estanterías, línea de cajas, salida. Una señal de ocultación
  cobra sentido combinada con "no pasó por caja" / "cruzó la salida".
- **API en FastAPI** que expone las señales y sirve las evidencias.
- **Panel de revisión en React + TypeScript** (tu terreno): cola de señales,
  reproducción del recorte, botones "validar / descartar", métricas en vivo,
  umbral configurable. Aquí el human-in-the-loop se hace tangible.

**Criterio de "hecho".** Una persona puede revisar una cola de señales, validar
o descartar, y esas decisiones quedan registradas (para auditar el sistema).

---

## Fase 5 — Tiempo real y despliegue

**Objetivo.** Pasar de vídeo grabado a flujo en vivo y medir rendimiento.

**Tareas.**
- Ingesta **RTSP** de cámara IP (varias cámaras → varios workers).
- Optimización: modelo más ligero, exportar a ONNX/TensorRT, considerar
  despliegue *edge* (p. ej. Jetson) frente a servidor central.
- Contenerización (Docker) y métricas de FPS / latencia por cámara.

**Criterio de "hecho".** El sistema procesa al menos una cámara en vivo a un FPS
aceptable, con cifras de rendimiento documentadas.

---

## Fase 6 — Evaluación, sesgos y documentación final

**Objetivo.** Cerrar el proyecto con rigor.

**Tareas.**
- Evaluación de **sesgos**: ¿el modelo se comporta peor con cierta ropa,
  complexión, iluminación? Documenta y mitiga.
- DPIA completa y "model card" del clasificador (datos, métricas, límites).
- Guía de despliegue responsable y checklist legal.

**Criterio de "hecho".** Cualquiera que lea el repo entiende qué hace, qué no
hace, cómo de bueno es y qué precauciones exige.

---

## Cómo presentar esto en una entrevista

Tres mensajes que dejan buena impresión:

1. **Acotaste el problema**: no prometiste "detectar hurtos" sino "señalar un
   gesto concreto para revisión humana". Eso es criterio de ingeniería.
2. **Pensaste en los falsos positivos y en lo legal** antes que en el modelo.
3. **Diseñaste para evolucionar**: la heurística es reemplazable por ML sin
   reescribir el pipeline, y lo demuestras con tests.

---

## Orden sugerido de trabajo

```
Fase 0 (medio día)  ->  Fase 1 (ya hecha)  ->  Fase 2 (1 semana)
   ->  Fase 3 (2-3 semanas, el grueso)  ->  Fase 4 (1-2 semanas)
   ->  Fase 5 (opcional, según ambición)  ->  Fase 6 (cierre)
```

No hace falta llegar a la Fase 5 para tener un portfolio sólido. Un proyecto
bien rematado hasta la Fase 4, con su panel de revisión y su comparación
heurística-vs-modelo, ya cuenta una historia técnica completa.
