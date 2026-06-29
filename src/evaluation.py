"""Evaluacion de la deteccion: precision / recall a nivel de evento (Fase 2).

Para calibrar los umbrales con criterio (no a ojo) hace falta comparar lo que
dispara el detector contra una verdad-terreno etiquetada a mano. Este modulo es
PURA logica: no abre video ni usa el modelo, asi que se testea con datos
sinteticos.

Modelo de evaluacion (deteccion de eventos en el tiempo):
- La verdad-terreno son INTERVALOS positivos por video: (video, inicio, fin) en
  segundos. Un video sin intervalos es 100% negativo.
- Una prediccion es un INSTANTE (segundos) en que el detector disparo.
- Una prediccion es acierto (TP) si cae dentro de algun intervalo positivo (con
  una tolerancia opcional). Si no, es falsa alarma (FP).
- Un intervalo positivo se considera detectado si al menos una prediccion cae
  dentro; los intervalos sin ninguna prediccion son fallos (FN).

  recall    = intervalos detectados / intervalos positivos totales
  precision = predicciones acertadas / predicciones totales
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class GTInterval:
    """Intervalo positivo etiquetado a mano (segundos)."""

    video: str
    start: float
    end: float


@dataclass(frozen=True)
class Prediction:
    """Un disparo del detector en un instante (segundos)."""

    video: str
    t: float


@dataclass
class EvalResult:
    tp_intervals: int   # intervalos positivos detectados (>=1 prediccion dentro)
    fn_intervals: int   # intervalos positivos no detectados
    tp_preds: int       # predicciones que caen dentro de algun intervalo
    fp_preds: int       # predicciones fuera de todo intervalo (falsas alarmas)

    @property
    def recall(self) -> float:
        total = self.tp_intervals + self.fn_intervals
        return self.tp_intervals / total if total else 0.0

    @property
    def precision(self) -> float:
        total = self.tp_preds + self.fp_preds
        return self.tp_preds / total if total else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) else 0.0


def _matches(t: float, iv: GTInterval, tol: float) -> bool:
    return iv.start - tol <= t <= iv.end + tol


def evaluate(ground_truth: list[GTInterval],
             predictions: list[Prediction],
             tol: float = 0.0) -> EvalResult:
    """Compara predicciones contra la verdad-terreno y devuelve las metricas.

    tol: margen en segundos para considerar que una prediccion cae dentro de un
    intervalo (el gesto dura un rato; el disparo es puntual).
    """
    # Recall: cada intervalo, ¿tiene alguna prediccion dentro?
    tp_intervals = 0
    for iv in ground_truth:
        if any(p.video == iv.video and _matches(p.t, iv, tol) for p in predictions):
            tp_intervals += 1
    fn_intervals = len(ground_truth) - tp_intervals

    # Precision: cada prediccion, ¿cae dentro de algun intervalo de su video?
    tp_preds = 0
    for p in predictions:
        if any(iv.video == p.video and _matches(p.t, iv, tol) for iv in ground_truth):
            tp_preds += 1
    fp_preds = len(predictions) - tp_preds

    return EvalResult(tp_intervals=tp_intervals, fn_intervals=fn_intervals,
                      tp_preds=tp_preds, fp_preds=fp_preds)
