"""
peak_detection.py
Detección de picos R, ondas P, complejos QRS, BPM y diagnóstico de bloqueos AV.

Funciones exportadas:
- detect_r_peaks()         : detecta picos R usando umbral y distancia mínima
- calculate_bpm()          : calcula BPM mediante mediana robusta de intervalos RR
- detect_qrs_complex()     : localiza inicio, pico y fin de cada complejo QRS
- detect_p_waves()         : localiza ondas P y calcula intervalos PR (ms)
- classify_rhythm()        : clasifica el ritmo por BPM (NORMAL/BRADYCARDIA/TACHYCARDIA/ASYSTOLE)
- analyze_av_blocks()      : clasifica bloqueos AV (1°, 2°-Mobitz I, 2°-Mobitz II, 3°)
- analyze_cardiac_cycle()  : análisis integral con decisión de marcapasos

Notas clínicas:
- BAV 1°  : PR > 200 ms con relación P:QRS 1:1
- BAV 2° Mobitz I (Wenckebach): PR se alarga progresivamente hasta perder un QRS
- BAV 2° Mobitz II: PR estable pero ondas P sin QRS asociado
- BAV 3° : disociación AV total (PR caótico, P>>R)
"""

from __future__ import annotations
import numpy as np

from . import config


# =========================================================
# ---------------- CÓDIGOS DE DIAGNÓSTICO -----------------
# =========================================================

DX_NORMAL          = "NORMAL"
DX_BAV_1           = "BAV_1"
DX_BAV_2_MOBITZ_I  = "BAV_2_MOBITZ_I"
DX_BAV_2_MOBITZ_II = "BAV_2_MOBITZ_II"
DX_BAV_3           = "BAV_3"
DX_ASYSTOLE        = "ASYSTOLE"
DX_INSUFFICIENT    = "INSUFFICIENT"

# Etiquetas legibles (es-CO) para la UI
AV_BLOCK_LABELS = {
    DX_NORMAL:          "SIN BLOQUEO",
    DX_BAV_1:           "BAV 1°",
    DX_BAV_2_MOBITZ_I:  "BAV 2° MOBITZ I",
    DX_BAV_2_MOBITZ_II: "BAV 2° MOBITZ II",
    DX_BAV_3:           "BAV 3°",
    DX_ASYSTOLE:        "ASISTOLIA",
    DX_INSUFFICIENT:    "---",
}

# Diagnósticos que justifican estimulación automática (riesgo vital)
AV_RISK_CRITICAL = (DX_BAV_2_MOBITZ_II, DX_BAV_3, DX_ASYSTOLE)


# =========================================================
# ---------------- UTILIDADES INTERNAS --------------------
# =========================================================

def _moving_average(x: np.ndarray, n: int) -> np.ndarray:
    """Suavizado por media móvil para reducir ruido antes de buscar picos."""
    if n <= 1:
        return x
    kernel = np.ones(int(n), dtype=float) / float(n)
    return np.convolve(x, kernel, mode="same")


# =========================================================
# ---------------- DETECCIÓN DE PICOS R -------------------
# =========================================================

def detect_r_peaks(signal_data, threshold: float, distance: int):
    """
    Detecta picos R en la señal ECG.

    Usa el valor absoluto de la señal para soportar QRS positivo y negativo.
    Aplica suavizado leve para eliminar falsos picos por ruido.
    """
    x = np.asarray(signal_data, dtype=float)
    if x.size < 3:
        return []

    distance = int(max(1, distance))
    x_abs = np.abs(x)
    x_f   = _moving_average(x_abs, n=5)

    thr       = float(threshold)
    peaks     = []
    last_peak = -distance

    for i in range(1, len(x_f) - 1):
        if x_f[i] > thr and x_f[i] > x_f[i - 1] and x_f[i] > x_f[i + 1]:
            if i - last_peak >= distance:
                peaks.append(i)
                last_peak = i

    return peaks


# =========================================================
# ---------------- CÁLCULO DE BPM -------------------------
# =========================================================

def calculate_bpm(peaks, sample_rate: float):
    """BPM via mediana de intervalos RR, descartando intervalos imposibles."""
    if len(peaks) < 2:
        return 0.0

    sr = float(sample_rate)
    rr = np.diff(np.asarray(peaks, dtype=float)) / sr
    rr = rr[(rr >= 0.25) & (rr <= 2.0)]

    if rr.size == 0:
        return 0.0

    bpm = 60.0 / float(np.median(rr))
    if bpm < 30 or bpm > 240:
        return 0.0

    return round(bpm, 1)


# =========================================================
# ---------------- DETECCIÓN DEL COMPLEJO QRS -------------
# =========================================================

def detect_qrs_complex(signal_data, r_peaks, sample_rate: float):
    """Localiza onset, pico y offset de cada complejo QRS (ventana 60 ms)."""
    x  = np.asarray(signal_data, dtype=float)
    n  = len(x)
    sr = float(sample_rate)

    search_before = int(sr * 0.060)
    search_after  = int(sr * 0.060)

    qrs_list = []

    for peak in r_peaks:
        if peak < 0 or peak >= n:
            continue

        onset    = max(0, peak - search_before)
        peak_abs = abs(x[peak])
        for i in range(peak - 1, max(0, peak - search_before) - 1, -1):
            if peak_abs > 0 and abs(x[i]) < peak_abs * 0.15:
                onset = i
                break

        offset = min(n - 1, peak + search_after)
        for i in range(peak + 1, min(n, peak + search_after + 1)):
            if peak_abs > 0 and abs(x[i]) < peak_abs * 0.15:
                offset = i
                break

        qrs_list.append({"onset": onset, "peak": peak, "offset": offset})

    return qrs_list


# =========================================================
# ---------------- DETECCIÓN DE ONDA P --------------------
# =========================================================

def detect_p_waves(signal_data, qrs_list, sample_rate: float):
    """
    Encuentra ondas P y calcula intervalos PR en milisegundos.

    Estrategia en dos pasadas:
      1) Asociación: por cada QRS retrocede 350 ms y toma el máximo local
         como P asociada al latido. Registra el PR correspondiente.
      2) Búsqueda de P 'extra': escanea entre QRS consecutivos para detectar
         ondas P no conducidas (criterio clave para BAV 2° y 3°).

    Retorna (p_peaks_idx, pr_intervals_ms).
    """
    x  = np.asarray(signal_data, dtype=float)
    sr = float(sample_rate)

    if not qrs_list or x.size < int(sr * 0.4):
        return [], []

    pr_window_sa = int(sr * 0.350)   # ventana PR fisiológica
    p_peaks        = []
    pr_intervals   = []

    # ── Pasada 1: P asociada a cada QRS ──────────────────────────
    for qrs in qrs_list:
        onset = int(qrs.get("onset", qrs.get("peak", 0)))
        start = max(0, onset - pr_window_sa)
        if start >= onset - 2:
            continue

        window = x[start:onset]
        if window.size == 0:
            continue

        smoothed = _moving_average(window, 5) if window.size >= 5 else window
        p_local  = int(np.argmax(smoothed))
        p_global = start + p_local
        p_peaks.append(p_global)

        pr_ms = (int(qrs.get("peak", onset)) - p_global) / sr * 1000.0
        pr_intervals.append(float(pr_ms))

    # ── Pasada 2: P 'extra' entre QRS (latidos bloqueados) ───────
    if len(qrs_list) >= 2 and p_peaks:
        baseline   = float(np.median(x))
        assoc_amps = [abs(x[i] - baseline) for i in p_peaks if 0 <= i < x.size]
        p_thr      = float(np.median(assoc_amps)) * 0.5 if assoc_amps else 0.0

        if p_thr > 0:
            min_dist_sa = int(sr * 0.250)

            for i in range(len(qrs_list) - 1):
                seg_a = int(qrs_list[i].get("offset", 0))     + int(sr * 0.180)
                seg_b = int(qrs_list[i + 1].get("onset", 0))  - int(sr * 0.050)
                if seg_b - seg_a < int(sr * 0.080):
                    continue

                segment = x[seg_a:seg_b]
                if segment.size < 5:
                    continue

                smoothed = _moving_average(segment, 5)
                center   = int(np.argmax(smoothed))
                val      = float(smoothed[center])

                if (val - baseline) >= p_thr:
                    cand = seg_a + center
                    if all(abs(cand - p) >= min_dist_sa for p in p_peaks):
                        p_peaks.append(cand)

    return sorted(p_peaks), pr_intervals


# =========================================================
# ---------------- CLASIFICACIÓN DE RITMO -----------------
# =========================================================

def classify_rhythm(bpm: float) -> str:
    """Clasifica por BPM: ASYSTOLE / BRADYCARDIA / NORMAL / TACHYCARDIA."""
    if bpm <= 0:
        return "ASYSTOLE"
    if bpm < 60:
        return "BRADYCARDIA"
    if bpm > 100:
        return "TACHYCARDIA"
    return "NORMAL"


# =========================================================
# ---------------- ANÁLISIS DE BLOQUEOS AV ----------------
# =========================================================

def analyze_av_blocks(p_peaks, r_peaks, pr_intervals_ms) -> str:
    """
    Clasifica bloqueos AV mediante análisis estadístico de PR y razón P:R.

    Orden de evaluación (más severo → menos severo):
      1) BAV 3° : disociación AV — PR caótico (std > 50 ms) y P >> R
      2) BAV 2° Mobitz I : alargamiento progresivo (diff > 15 ms) y P > R
      3) BAV 2° Mobitz II: PR estable (std ≤ 20 ms) con P > R
      4) BAV 1° : PR medio > 200 ms con conducción 1:1
      5) NORMAL
    """
    n_r = len(r_peaks)
    n_p = len(p_peaks)

    if n_r == 0:
        return DX_ASYSTOLE
    if n_r < 2:
        return DX_INSUFFICIENT

    min_pr = int(getattr(config, "AV_MIN_PR_FOR_DIAGNOSIS", 3))
    if len(pr_intervals_ms) < min_pr:
        return DX_NORMAL

    pr      = np.asarray(pr_intervals_ms, dtype=float)
    pr_std  = float(np.std(pr))
    pr_mean = float(np.mean(pr))
    pr_diff = np.diff(pr)

    std_chaotic     = float(getattr(config, "AV_PR_CHAOTIC_STD_MS",      50.0))
    diff_progress   = float(getattr(config, "AV_PR_PROGRESSIVE_DIFF_MS", 15.0))
    std_stable      = float(getattr(config, "AV_PR_STABLE_STD_MS",       20.0))
    pr_normal_max   = float(getattr(config, "AV_PR_NORMAL_MAX_MS",      200.0))

    # 1) Disociación AV (BAV 3°)
    if pr_std > std_chaotic and n_p > n_r:
        return DX_BAV_3

    # 2) Mobitz I (alargamiento progresivo)
    if pr_diff.size > 0 and np.any(pr_diff > diff_progress) and n_p > n_r:
        return DX_BAV_2_MOBITZ_I

    # 3) Mobitz II (PR estable + P bloqueada)
    if pr_std <= std_stable and n_p > n_r:
        return DX_BAV_2_MOBITZ_II

    # 4) BAV 1° (PR alargado, conducción 1:1)
    if pr_mean > pr_normal_max and n_p == n_r:
        return DX_BAV_1

    return DX_NORMAL


# =========================================================
# ---------------- ANÁLISIS INTEGRAL ----------------------
# =========================================================

def analyze_cardiac_cycle(peaks, signal_data, sample_rate,
                          min_bpm: float = 50.0,
                          max_rr_interval: float = 2.0) -> dict:
    """
    Análisis cardiaco integral.

    Retorna dict con:
      bpm, rhythm, diagnosis, asystole, bradycardia, pacemaker_needed,
      p_peaks, pr_intervals_ms, qrs, last_rr_interval
    """
    status = {
        "bpm":              0.0,
        "rhythm":           "ASYSTOLE",
        "diagnosis":        DX_INSUFFICIENT,
        "asystole":         False,
        "bradycardia":      False,
        "pacemaker_needed": False,
        "p_peaks":          [],
        "pr_intervals_ms":  [],
        "qrs":              [],
        "last_rr_interval": None,
    }

    if len(peaks) < 2:
        status["asystole"]         = True
        status["pacemaker_needed"] = True
        status["diagnosis"]        = DX_ASYSTOLE
        status["rhythm"]           = "ASYSTOLE"
        return status

    sr           = float(sample_rate)
    rr_intervals = np.diff(peaks) / sr
    last_rr      = float(rr_intervals[-1])
    median_rr    = float(np.median(rr_intervals))
    bpm          = (60.0 / median_rr) if median_rr > 0 else 0.0

    status["bpm"]              = round(bpm, 1)
    status["last_rr_interval"] = last_rr
    status["rhythm"]           = classify_rhythm(bpm)

    # Detección morfológica
    qrs_list                = detect_qrs_complex(signal_data, peaks, sr)
    p_peaks, pr_intervals   = detect_p_waves(signal_data, qrs_list, sr)
    diagnosis               = analyze_av_blocks(p_peaks, peaks, pr_intervals)

    status["qrs"]             = qrs_list
    status["p_peaks"]         = p_peaks
    status["pr_intervals_ms"] = pr_intervals
    status["diagnosis"]       = diagnosis

    # ── Lógica de activación del marcapasos (riesgo vital) ──────
    crit_bpm = float(getattr(config, "AV_CRITICAL_LOW_BPM", 40.0))

    # Asistolia funcional (último RR demasiado largo)
    if last_rr > max_rr_interval:
        status["asystole"]         = True
        status["pacemaker_needed"] = True

    # Bloqueos AV de alto grado → marcapasos
    if diagnosis in AV_RISK_CRITICAL:
        status["pacemaker_needed"] = True

    # Mobitz I con bradicardia significativa → marcapasos
    if bpm > 0 and bpm < min_bpm and diagnosis == DX_BAV_2_MOBITZ_I:
        status["pacemaker_needed"] = True

    # Bradicardia crítica absoluta
    if 0 < bpm < crit_bpm:
        status["bradycardia"]      = True
        status["pacemaker_needed"] = True

    if bpm < min_bpm and bpm > 0:
        status["bradycardia"] = True

    return status
