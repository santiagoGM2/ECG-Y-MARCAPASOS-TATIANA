"""
peak_detection.py
Deteccion de picos R, calculo de BPM y analisis del ritmo cardiaco.

Funciones exportadas:
- detect_r_peaks()    : detecta picos R usando umbral y distancia minima
- calculate_bpm()     : calcula BPM mediante mediana robusta de intervalos RR
- detect_qrs_complex(): localiza inicio, pico y fin de cada complejo QRS
- classify_rhythm()   : clasifica el ritmo como NORMAL/BRADYCARDIA/TACHYCARDIA/ASYSTOLE
- analyze_cardiac_cycle(): analisis completo del ciclo cardiaco (compatibilidad)
"""

from __future__ import annotations
import numpy as np


# =========================================================
# ---------------- UTILIDADES INTERNAS --------------------
# =========================================================

def _moving_average(x: np.ndarray, n: int) -> np.ndarray:
    """Suavizado por media movil para reducir ruido antes de buscar picos."""
    if n <= 1:
        return x
    kernel = np.ones(int(n), dtype=float) / float(n)
    return np.convolve(x, kernel, mode="same")


# =========================================================
# ---------------- DETECCION DE PICOS R -------------------
# =========================================================

def detect_r_peaks(signal_data, threshold: float, distance: int):
    """
    Detecta picos R en la señal ECG.

    Usa el valor absoluto de la señal para soportar QRS positivo y negativo.
    Aplica suavizado leve para eliminar falsos picos por ruido.

    Args:
        signal_data : array-like de voltajes (centrado en 0)
        threshold   : umbral minimo sobre |señal| para detectar un pico (Voltios)
        distance    : distancia minima entre picos R consecutivos (samples)

    Returns:
        list[int] : indices de los picos R detectados dentro de la ventana
    """
    x = np.asarray(signal_data, dtype=float)
    if x.size < 3:
        return []

    distance = int(max(1, distance))

    # Valor absoluto: funciona para QRS positivo y negativo
    x_abs = np.abs(x)

    # Suavizado leve de 5 samples para reducir ruido de alta frecuencia
    x_f = _moving_average(x_abs, n=5)

    thr    = float(threshold)
    peaks  = []
    last_peak = -distance

    for i in range(1, len(x_f) - 1):
        if x_f[i] > thr and x_f[i] > x_f[i - 1] and x_f[i] > x_f[i + 1]:
            if i - last_peak >= distance:
                peaks.append(i)
                last_peak = i

    return peaks


# =========================================================
# ---------------- CALCULO DE BPM -------------------------
# =========================================================

def calculate_bpm(peaks, sample_rate: float):
    """
    Calcula la frecuencia cardiaca (BPM) usando la mediana de intervalos RR.

    Descarta intervalos fisiologicamente imposibles para mayor robustez.

    Args:
        peaks       : lista de indices de picos R detectados
        sample_rate : frecuencia de muestreo en Hz

    Returns:
        float : BPM calculado, 0.0 si no hay suficientes datos validos
    """
    if len(peaks) < 2:
        return 0.0

    sr = float(sample_rate)
    rr = np.diff(np.asarray(peaks, dtype=float)) / sr  # intervalos en segundos

    # Filtrar intervalos fisiologicamente validos: 30-240 BPM
    rr = rr[(rr >= 0.25) & (rr <= 2.0)]

    if rr.size == 0:
        return 0.0

    bpm = 60.0 / float(np.median(rr))

    if bpm < 30 or bpm > 240:
        return 0.0

    return round(bpm, 1)


# =========================================================
# ---------------- DETECCION DEL COMPLEJO QRS -------------
# =========================================================

def detect_qrs_complex(signal_data, r_peaks, sample_rate: float):
    """
    Localiza el inicio (onset), pico y fin (offset) de cada complejo QRS.

    Busca el cruce por cero o minimo local mas cercano al pico R
    en una ventana de 60ms antes y despues del pico.

    Args:
        signal_data : array de voltajes de la señal ECG
        r_peaks     : lista de indices de picos R detectados
        sample_rate : frecuencia de muestreo en Hz

    Returns:
        list[dict] : cada dict contiene {'onset': int, 'peak': int, 'offset': int}
    """
    x  = np.asarray(signal_data, dtype=float)
    n  = len(x)
    sr = float(sample_rate)

    # Ventana de busqueda: 60ms antes y despues del pico R
    search_before = int(sr * 0.060)
    search_after  = int(sr * 0.060)

    qrs_list = []

    for peak in r_peaks:
        if peak < 0 or peak >= n:
            continue

        # --- Buscar onset: retroceder hasta valor bajo o cero ---
        onset = max(0, peak - search_before)
        peak_abs = abs(x[peak])
        for i in range(peak - 1, max(0, peak - search_before) - 1, -1):
            if peak_abs > 0 and abs(x[i]) < peak_abs * 0.15:
                onset = i
                break

        # --- Buscar offset: avanzar hasta valor bajo o cero ---
        offset = min(n - 1, peak + search_after)
        for i in range(peak + 1, min(n, peak + search_after + 1)):
            if peak_abs > 0 and abs(x[i]) < peak_abs * 0.15:
                offset = i
                break

        qrs_list.append({
            "onset":  onset,
            "peak":   peak,
            "offset": offset,
        })

    return qrs_list


# =========================================================
# ---------------- CLASIFICACION DEL RITMO ----------------
# =========================================================

def classify_rhythm(bpm: float) -> str:
    """
    Clasifica el ritmo cardiaco segun la frecuencia cardiaca.

    Criterios clinicos estandar:
    - ASYSTOLE    : BPM = 0 (sin latidos detectados)
    - BRADYCARDIA : BPM < 60 latidos por minuto
    - NORMAL      : 60 <= BPM <= 100
    - TACHYCARDIA : BPM > 100 latidos por minuto

    Args:
        bpm : frecuencia cardiaca calculada en BPM

    Returns:
        str : clasificacion del ritmo
    """
    if bpm <= 0:
        return "ASYSTOLE"
    elif bpm < 60:
        return "BRADYCARDIA"
    elif bpm > 100:
        return "TACHYCARDIA"
    else:
        return "NORMAL"


# =========================================================
# ---------------- ANALISIS DEL CICLO CARDIACO ------------
# =========================================================

def analyze_cardiac_cycle(peaks, sample_rate, min_bpm=50, max_rr_interval=2.0):
    """
    Analiza el ciclo cardiaco y detecta condiciones de emergencia.
    Mantenida por compatibilidad con versiones anteriores.

    Args:
        peaks           : lista de indices de picos R
        sample_rate     : frecuencia de muestreo en Hz
        min_bpm         : BPM minimo seguro (default 50)
        max_rr_interval : intervalo RR maximo en segundos (default 2.0)

    Returns:
        dict : estado cardiaco con llaves bpm, asystole, bradycardia, pacemaker_needed
    """
    status = {
        "bpm":               0,
        "asystole":          False,
        "bradycardia":       False,
        "pacemaker_needed":  False,
        "last_rr_interval":  None,
    }

    if len(peaks) < 2:
        status["asystole"]         = True
        status["pacemaker_needed"] = True
        return status

    rr_intervals = np.diff(peaks) / float(sample_rate)
    last_rr      = float(rr_intervals[-1])
    bpm          = 60.0 / float(np.median(rr_intervals))

    status["bpm"]              = bpm
    status["last_rr_interval"] = last_rr

    if last_rr > max_rr_interval:
        status["asystole"]         = True
        status["pacemaker_needed"] = True
    elif bpm < min_bpm:
        status["bradycardia"]      = True
        status["pacemaker_needed"] = True

    return status
