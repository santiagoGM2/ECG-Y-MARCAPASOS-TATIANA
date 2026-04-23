"""
config.py
Configuracion global del sistema ECG + Marcapasos.
Hardware: ESP32 + CD4051 MUX. Interfaz Python/tkinter.
"""

# =========================================================
# ---------------- SERIAL CONFIGURATION -------------------
# =========================================================

# Puerto del ESP32 (cambiar segun el sistema)
SERIAL_PORT = "COM6"

# Baudrate (debe coincidir con el firmware del ESP32)
BAUDRATE = 115200

# Timeout de lectura serial (segundos)
SERIAL_TIMEOUT = 1

# =========================================================
# ---------------- SAMPLING CONFIG ------------------------
# =========================================================

# Frecuencia de muestreo del ECG (Hz) — debe coincidir con el ADC del ESP32
SAMPLE_RATE = 1000

# Habilitar/deshabilitar filtros de señal
ENABLE_FILTERS = False

# Intervalo de actualizacion de la GUI (ms)
REFRESH_INTERVAL = 80

# Tamaño maximo del buffer circular
MAX_BUFFER_SIZE = 5000

# =========================================================
# ---------------- MUX CONFIGURATION ----------------------
# =========================================================

# Prefijo del comando enviado al ESP32
MUX_COMMAND_PREFIX = "MUX:"

# Numero total de derivadas del ECG
TOTAL_DERIVATIONS = 6

# Formato del comando MUX enviado al firmware
MUX_COMMAND_FORMAT = "STATE_{state}\n"

# =========================================================
# ---------------- SIGNAL DISPLAY CONFIG ------------------
# =========================================================

# Valor maximo inicial del eje Y (Voltios)
DEFAULT_Y_MAX = 2.0

# Tamaño inicial de la ventana de visualizacion (samples)
DEFAULT_WINDOW_SIZE = 2000

# Ganancia inicial de visualizacion
DEFAULT_GAIN = 1.0

# =========================================================
# ---------------- PEAK DETECTION CONFIG ------------------
# =========================================================

# Umbral inicial para deteccion de pico R (Voltios)
DEFAULT_R_THRESHOLD = 0.3

# Distancia minima entre picos R consecutivos (samples)
DEFAULT_R_DISTANCE = 200

# =========================================================
# ---------------- SYSTEM MODES ---------------------------
# =========================================================

MODE_MANUAL = "MANUAL"
MODE_AUTO   = "AUTO"

# Segundos sin interaccion antes de pasar a modo AUTO
AUTO_TIMEOUT = 10

# Segundos entre cambios de derivada en modo AUTO
AUTO_SWITCH_INTERVAL = 8

# =========================================================
# ---------------- DEBUG ----------------------------------
# =========================================================

ENABLE_DEBUG_PRINTS = False

# =========================================================
# ---------------- PACEMAKER CONFIG -----------------------
# =========================================================

# Tiempo de alerta visual del marcapasos (segundos)
PACE_UI_ALERT_SEC    = 1.5
PACE_ALERT_HOLD_SEC  = PACE_UI_ALERT_SEC
PACE_TIMEOUT_SEC     = 1.5
PACE_MIN_INTERVAL_SEC = 0.8

# Amplitud del spike de marcapasos (Voltios)
PACE_SPIKE_AMPLITUDE = 1.0

# Ancho del spike en samples (~4 ms a 1kHz)
PACE_SPIKE_WIDTH_SAMPLES = 4

# Duracion del pulso bifasico (ms)
PACE_SPIKE_DURATION_MS = 4

# Umbral de derivada para detectar spike automaticamente
PACE_DERIV_THRESHOLD = 0.6

# =========================================================
# ---------------- DISPLAY / UX ---------------------------
# =========================================================

# Blanking despues de cambiar derivada (segundos)
DERIVATION_SWITCH_BLANK_SEC = 2.5
BLANK_AFTER_SWITCH_SEC      = DERIVATION_SWITCH_BLANK_SEC

# =========================================================
# ---------------- NO SIGNAL DETECTION --------------------
# =========================================================

# Ventana de muestras para calcular P2P y decidir si hay señal
NO_SIGNAL_WINDOW_SAMPLES = 500

# Umbral pico a pico minimo para considerar que hay señal (Voltios)
NO_SIGNAL_P2P_V         = 0.02
NO_SIGNAL_P2P_THRESHOLD = NO_SIGNAL_P2P_V

# Umbral de desviacion estandar para detectar señal
NO_SIGNAL_STD_V = 0.005

# Tiempo sin señal para mostrar baseline muerta (segundos)
NO_SIGNAL_TIMEOUT_SEC  = 1.0
NO_SIGNAL_MIN_SECONDS  = NO_SIGNAL_TIMEOUT_SEC

# =========================================================
# ---------------- SERIAL ROBUSTNESS ----------------------
# =========================================================

# Intervalo de reconexion serial (segundos)
SERIAL_RECONNECT_SEC = 3.0

# Tiempo maximo sin datos antes de considerar desconexion
SERIAL_STALE_SEC = 2.0

# =========================================================
# ---------------- ADC CONVERSION -------------------------
# =========================================================

# Voltaje de referencia del ADC del ESP32
ADC_VREF = 3.3

# Valor maximo del ADC de 12 bits
ADC_MAX = 4095

# =========================================================
# ---------------- SIMULATION MODE ------------------------
# =========================================================

# Habilitar modo simulacion (se sobreescribe en runtime segun hardware)
SIMULATION_MODE = True

# Frecuencia cardiaca de la simulacion (BPM)
SIMULATION_HEART_RATE = 72

# Desviacion estandar del ruido gaussiano en simulacion (mV)
SIMULATION_NOISE = 0.02

# =========================================================
# ---------------- SESSION INFO ---------------------------
# =========================================================

SESSION_LABEL = "BIOMEDICAL DEMO"
