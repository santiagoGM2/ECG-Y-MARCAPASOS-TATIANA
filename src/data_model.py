"""
data_model.py
Estado compartido del sistema ECG + Marcapasos.

Contiene:
- Buffers de señal (voltaje y tiempo)
- Estado de conexion del ESP32 y modo simulacion
- Estado del MUX (derivada activa)
- Modo MANUAL/AUTO y temporizadores
- Estados de UX: blanking, sin señal, marcapasos
- Variables de control del marcapasos (BPM, amplitud)
- Metricas cardiacas acumuladas
"""

import threading
import time
from collections import deque
import tkinter as tk
from . import config


class AppState:
    def __init__(self, master=None):

        # =====================================================
        # ---------------- DATA BUFFERS -----------------------
        # =====================================================
        self.data_lock = threading.Lock()

        maxlen = getattr(config, "MAX_BUFFER_SIZE", 5000)

        # Buffer principal de señal en Voltios
        self.voltage_buffer = deque(maxlen=maxlen)

        # Buffer de tiempo (indice de muestra)
        self.time_buffer = deque(maxlen=maxlen)

        # Alias de compatibilidad
        self.signal_buffer = self.voltage_buffer

        # Contador total de muestras recibidas
        self.sample_count = 0

        # =====================================================
        # --------------- CONNECTION STATUS -------------------
        # =====================================================
        # True si el ESP32 esta conectado y enviando datos
        self.esp32_connected = False

        # Alias de compatibilidad
        self.serial_connected = False

        # True cuando no hay hardware y se usa la simulacion
        self.simulation_mode = True

        # =====================================================
        # --------------- DERIVATION CONTROL ------------------
        # =====================================================
        # Mapa de estado MUX → etiqueta de derivada ECG
        self.mux_state_label = {
            0: "I",
            1: "II",
            2: "III",
            3: "aVR",
            4: "aVL",
            5: "aVF",
        }

        # Derivada activa (default: Lead II, mas usada en clinica)
        self.current_mux_state = 1
        self.mux_lock = threading.Lock()

        # =====================================================
        # ----------- UX STATES (blanking / no signal / pace) -
        # =====================================================
        # Hasta este timestamp se dibuja baseline muerta (blanking)
        self.blank_until = 0.0

        # Bandera de ausencia de señal
        self.no_signal = False
        self.no_signal_since = None

        # Marcapasos por UI: True cuando el usuario presiona el boton
        self.pace_pulse_pending = False

        # Timestamp hasta el que se muestra la alerta de marcapasos
        self.pace_alert_until = 0.0

        # =====================================================
        # ----------- MANUAL / AUTO CONTROL MODE -------------
        # =====================================================
        self.operation_mode = tk.StringVar(
            master=master, value=config.MODE_MANUAL
        )

        self.last_manual_action_time = time.time()
        self.last_auto_switch_time   = time.time()

        # =====================================================
        # -------- UI VARIABLES (AFECTAN EL PROCESAMIENTO) ----
        # =====================================================
        # Umbral para deteccion de pico R (Voltios)
        self.r_threshold = tk.DoubleVar(
            master=master, value=config.DEFAULT_R_THRESHOLD
        )

        # Distancia minima entre picos R (samples)
        self.r_distance = tk.IntVar(
            master=master, value=config.DEFAULT_R_DISTANCE
        )

        # Ganancia de visualizacion de la señal
        self.ecg_gain = tk.DoubleVar(
            master=master, value=config.DEFAULT_GAIN
        )

        # Numero de muestras visibles en la ventana del grafico
        self.window_size = tk.IntVar(
            master=master, value=config.DEFAULT_WINDOW_SIZE
        )

        # Limite maximo del eje Y (Voltios)
        self.y_max = tk.DoubleVar(
            master=master, value=config.DEFAULT_Y_MAX
        )

        # =====================================================
        # -------- VARIABLES DEL MARCAPASOS -------------------
        # =====================================================
        # Frecuencia de estimulacion del marcapasos (BPM)
        self.pace_bpm_var = tk.DoubleVar(master=master, value=60.0)

        # Amplitud del pulso del marcapasos (Voltios)
        self.pace_amplitude_var = tk.DoubleVar(master=master, value=1.0)

        # =====================================================
        # -------- METRICAS CARDIACAS -------------------------
        # =====================================================
        # Total de complejos QRS detectados en la sesion
        self.qrs_detected_count = 0

        # Ultimo valor de BPM calculado
        self.last_bpm = 0.0

    # =========================================================
    # ---------------- SIGNAL ACCESS ---------------------------
    # =========================================================

    def get_current_signal(self):
        """Retorna lista de la señal en voltios (thread-safe)."""
        with self.data_lock:
            return list(self.voltage_buffer)

    def add_sample(self, volts: float):
        """Agrega una muestra en voltios de forma thread-safe."""
        with self.data_lock:
            self.voltage_buffer.append(float(volts))
            self.time_buffer.append(self.sample_count)
            self.sample_count += 1

    def add_samples_batch(self, voltages):
        """
        Agrega un lote de muestras de forma thread-safe.
        Optimiza la contención del lock (importante en Windows/Tkinter).
        """
        if not voltages:
            return
        with self.data_lock:
            start = int(self.sample_count)
            n = len(voltages)
            # Extend deques en bloque para minimizar overhead de Python
            self.voltage_buffer.extend(float(v) for v in voltages)
            self.time_buffer.extend(range(start, start + n))
            self.sample_count = start + n

    # =========================================================
    # ---------------- MUX CONTROL -----------------------------
    # =========================================================

    def set_mux_state(self, state: int):
        """Selecciona una derivada manualmente y resetea modo a MANUAL."""
        total = getattr(config, "TOTAL_DERIVATIONS", 6)
        with self.mux_lock:
            self.current_mux_state = int(state) % total
            self.operation_mode.set(config.MODE_MANUAL)
            self.last_manual_action_time = time.time()

    def next_derivation(self):
        """Avanza a la siguiente derivada en orden circular."""
        total = getattr(config, "TOTAL_DERIVATIONS", 6)
        with self.mux_lock:
            self.current_mux_state = (self.current_mux_state + 1) % total

    # =========================================================
    # ---------------- AUTO MODE LOGIC -------------------------
    # =========================================================

    def check_auto_mode(self):
        """Activa modo AUTO si no hay interaccion manual por AUTO_TIMEOUT segundos."""
        if time.time() - self.last_manual_action_time > config.AUTO_TIMEOUT:
            self.operation_mode.set(config.MODE_AUTO)

    def auto_switch_if_needed(self):
        """Cambia derivada automaticamente si ha pasado AUTO_SWITCH_INTERVAL."""
        if time.time() - self.last_auto_switch_time > config.AUTO_SWITCH_INTERVAL:
            self.next_derivation()
            self.last_auto_switch_time = time.time()
