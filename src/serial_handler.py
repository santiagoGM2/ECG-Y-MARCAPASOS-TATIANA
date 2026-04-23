"""
serial_handler.py
Manejador de comunicacion serial con el ESP32 y modo simulacion ECG.

Modos de operacion:
1. HARDWARE: Lee datos reales del ESP32 via USB serial.
   - Parser hibrido: paquetes binarios [0xAA][LSB][MSB][XOR] y ASCII.
   - Reconexion automatica cada SERIAL_RECONNECT_SEC segundos.
   - Comandos: MUX (cambio de derivada) y PACE (marcapasos).

2. SIMULACION: Activa automaticamente cuando no hay hardware disponible.
   - Genera señal ECG matematica con ondas P, QRS, T.
   - Configurable: BPM, amplitud, nivel de ruido.
   - Soporta marcapasos automatico con pulso bifasico.
   - Soporta simulacion de arritmias temporales.
"""

import time
import threading
import math
import numpy as np

try:
    import serial
    import serial.tools.list_ports
    SERIAL_AVAILABLE = True
except ImportError:
    SERIAL_AVAILABLE = False

from . import config


def list_available_ports():
    """Retorna lista de puertos COM disponibles en el sistema."""
    if not SERIAL_AVAILABLE:
        return []
    try:
        ports = serial.tools.list_ports.comports()
        return [p.device for p in sorted(ports, key=lambda p: p.device)]
    except Exception:
        return []


class SerialReader(threading.Thread):
    """
    Hilo de adquisicion de datos: hardware real o simulacion ECG matematica.

    Thread-safe: todo acceso a app_state.voltage_buffer usa data_lock.
    """

    PKT_START = 0xAA  # Byte de inicio del paquete binario

    def __init__(self, app_state):
        super().__init__(daemon=True)
        self.app_state  = app_state
        self.running    = True

        # ── Parametros serial ──────────────────────────────────────
        self.serial_port  = None
        self._rx_buf      = bytearray()
        self._reconnect_sec = float(getattr(config, "SERIAL_RECONNECT_SEC", 3.0))

        # Conversion ADC (12 bits) → Voltios
        self._vref    = float(getattr(config, "ADC_VREF", 3.3))
        self._adc_max = int(getattr(config, "ADC_MAX", 4095))

        # Formato del comando MUX enviado al firmware
        self._mux_cmd_fmt = getattr(config, "MUX_COMMAND_FORMAT", "STATE_{state}\n")

        # ── Parametros del simulador ECG ──────────────────────────
        # Controlados desde la UI de simulacion
        self.sim_heart_rate  = float(getattr(config, "SIMULATION_HEART_RATE", 72))
        self.sim_noise_level = float(getattr(config, "SIMULATION_NOISE", 0.02))
        self.sim_amplitude   = 1.0

        # Marcapasos automatico en simulacion
        self.auto_pacing_enabled = False
        self.pace_amplitude      = float(getattr(config, "PACE_SPIKE_AMPLITUDE", 1.0))
        self.pace_bpm            = 60.0

        # Arritmia simulada temporal
        self.sim_arrhythmia       = False
        self.sim_arrhythmia_until = 0.0
        self.sim_waveform_type    = "NORMAL"  # NORMAL | BRADYCARDIA | TACHYCARDIA

        # Estado interno del generador
        self._sim_sample_count = 0
        self._sim_cycle_buffer = None
        self._sim_cycle_idx    = 0
        self._sim_last_bpm     = 0.0

        # ── Intentar conexion con hardware ────────────────────────
        self._open_port()

    # ==============================================================
    # ── GESTION DE CONEXION SERIAL ────────────────────────────────
    # ==============================================================

    def _set_connected(self, ok: bool, simulation: bool = False):
        """Actualiza los flags de conexion en AppState de forma atomica."""
        self.app_state.esp32_connected  = bool(ok)
        self.app_state.serial_connected = bool(ok)
        self.app_state.simulation_mode  = bool(simulation)

    def _open_port(self):
        """Intenta abrir el puerto serial configurado en config.SERIAL_PORT."""
        port = str(getattr(config, "SERIAL_PORT", "") or "").strip()
        # Permite iniciar siempre en simulación y conectar solo cuando el usuario lo pida desde la UI
        if (not port) or port.upper().startswith(("NONE", "SIM", "AUTO")):
            self.serial_port = None
            self._set_connected(False, simulation=True)
            return

        if not SERIAL_AVAILABLE:
            print("[Serial] pyserial no disponible. Iniciando simulacion.")
            self._set_connected(False, simulation=True)
            return

        try:
            self.serial_port = serial.Serial(
                port     = port,
                baudrate = config.BAUDRATE,
                timeout  = getattr(config, "SERIAL_TIMEOUT", 1),
            )
            # Esperar reset del ESP32 al abrir el puerto
            time.sleep(0.3)

            try:
                self.serial_port.reset_input_buffer()
                self.serial_port.reset_output_buffer()
            except Exception:
                pass

            self._set_connected(True, simulation=False)
            print(f"[Serial] Conectado a {port} a {config.BAUDRATE} baud.")

        except Exception as e:
            print(f"[Serial] No se pudo abrir {port}: {e}")
            print("[Serial] Iniciando MODO SIMULACION automaticamente.")
            self.serial_port = None
            self._set_connected(False, simulation=True)

    def _close_port(self):
        """Cierra el puerto serial de forma segura."""
        try:
            if self.serial_port and getattr(self.serial_port, "is_open", False):
                self.serial_port.close()
        except Exception:
            pass
        self.serial_port = None

    # ==============================================================
    # ── SIMULACION ECG MATEMATICA ─────────────────────────────────
    # ==============================================================

    def _generate_ecg_cycle(self, bpm: float) -> np.ndarray:
        """
        Genera un ciclo completo de ECG usando modelos gaussianos/triangulares.

        Morfologia: onda P + complejo QRS (Q, R, S) + onda T.
        Las posiciones son fracciones del ciclo cardiaco (0.0 a 1.0).

        Parametros clinicos:
          - P : amplitud 0.15 mV, duracion ~80ms, a 200ms antes del QRS
          - Q : amplitud -0.10 mV, duracion ~20ms
          - R : amplitud +1.00 mV, duracion ~80ms  (pico principal)
          - S : amplitud -0.30 mV, duracion ~20ms
          - T : amplitud +0.30 mV, duracion ~160ms, a 200ms despues del QRS
        """
        sample_rate   = float(getattr(config, "SAMPLE_RATE", 1000))
        cycle_samples = max(10, int(sample_rate * 60.0 / bpm))
        t             = np.linspace(0.0, 1.0, cycle_samples, endpoint=False)
        amp           = float(self.sim_amplitude)

        ecg = np.zeros(cycle_samples, dtype=float)

        # Onda P: Gaussiana centrada en 20% del ciclo
        p_pos   = 0.20
        p_std   = 0.035
        ecg    += 0.15 * amp * np.exp(-((t - p_pos) ** 2) / (2 * p_std ** 2))

        # Onda Q: deflexion negativa antes del pico R
        q_pos   = 0.42
        q_std   = 0.011
        ecg    -= 0.10 * amp * np.exp(-((t - q_pos) ** 2) / (2 * q_std ** 2))

        # Onda R: pico positivo principal (complejo QRS)
        r_pos   = 0.45
        r_std   = 0.018
        ecg    += 1.00 * amp * np.exp(-((t - r_pos) ** 2) / (2 * r_std ** 2))

        # Onda S: deflexion negativa despues del pico R
        s_pos   = 0.48
        s_std   = 0.011
        ecg    -= 0.30 * amp * np.exp(-((t - s_pos) ** 2) / (2 * s_std ** 2))

        # Onda T: Gaussiana centrada en 65% del ciclo
        t_pos   = 0.65
        t_std   = 0.055
        ecg    += 0.30 * amp * np.exp(-((t - t_pos) ** 2) / (2 * t_std ** 2))

        return ecg

    def _get_ecg_cycle(self, bpm: float) -> np.ndarray:
        """Retorna el ciclo ECG pre-generado, regenerando si cambio el BPM."""
        if self._sim_cycle_buffer is None or abs(bpm - self._sim_last_bpm) > 0.5:
            self._sim_cycle_buffer = self._generate_ecg_cycle(bpm)
            self._sim_last_bpm     = bpm
            self._sim_cycle_idx    = 0
        return self._sim_cycle_buffer

    def _get_sim_bpm(self) -> float:
        """Retorna el BPM efectivo segun el tipo de forma de onda configurado."""
        wt = getattr(self, "sim_waveform_type", "NORMAL")

        # 1) BPM base según el preset seleccionado (o el valor configurable)
        if wt == "BRADYCARDIA":
            base = 42.0
        elif wt == "TACHYCARDIA":
            base = 132.0
        else:
            base = float(self.sim_heart_rate)

        # 2) Arritmia temporal: variación aleatoria sobre el BPM base
        if self.sim_arrhythmia and time.time() < self.sim_arrhythmia_until:
            noise = float(np.random.uniform(-20, 20))
            return max(30.0, min(200.0, base + noise))

        # Expiró la arritmia: limpiar flag para que la UI quede consistente
        self.sim_arrhythmia = False
        return max(30.0, min(200.0, base))

    def _simulate_loop(self):
        """
        Bucle principal del generador de señal ECG simulada.

        Genera muestras en lotes de 20 para minimizar overhead de Python.
        Mantiene la tasa de muestreo real mediante sleep adaptativo.
        """
        sample_rate    = float(getattr(config, "SAMPLE_RATE", 1000))
        batch_size     = 20                         # muestras por iteracion (20ms)
        batch_interval = batch_size / sample_rate   # segundos por batch

        # Posicion del pico R en el ciclo (fraccion)
        r_pos_fraction = 0.45

        # Duracion del spike bifasico del marcapasos: 20ms
        spike_total_s  = 0.020
        spike_total_sa = int(sample_rate * spike_total_s)
        spike_half_sa  = spike_total_sa // 2

        # Marcar simulacion activa
        self._set_connected(False, simulation=True)

        while self.running:
            t_start = time.perf_counter()

            bpm        = self._get_sim_bpm()
            ecg_cycle  = self._get_ecg_cycle(bpm)
            cycle_len  = len(ecg_cycle)
            r_pos_sa   = int(r_pos_fraction * cycle_len)  # indice del pico R en el ciclo

            noise_std  = float(self.sim_noise_level)
            pace_amp   = float(self.pace_amplitude)
            pacing     = bool(self.auto_pacing_enabled)

            batch_voltages = []

            for _ in range(batch_size):
                idx          = self._sim_cycle_idx % cycle_len
                samples_to_r = (r_pos_sa - idx) % cycle_len

                # Spike bifasico del marcapasos: 20ms antes del pico R
                if pacing and samples_to_r <= spike_total_sa:
                    phase = spike_total_sa - samples_to_r
                    if phase < spike_half_sa:
                        v = pace_amp    # Fase positiva
                    else:
                        v = -pace_amp   # Fase negativa
                else:
                    # Señal ECG matematica normal
                    v = float(ecg_cycle[idx])

                    # Wander de baseline: seno a 0.05 Hz, amplitud 0.05 mV
                    t_abs  = self._sim_sample_count / sample_rate
                    v     += 0.05 * math.sin(2.0 * math.pi * 0.05 * t_abs)

                    # Ruido gaussiano configurable
                    if noise_std > 0:
                        v += float(np.random.normal(0.0, noise_std))

                batch_voltages.append(v)
                self._sim_cycle_idx    += 1
                self._sim_sample_count += 1

            # Push thread-safe al AppState (en bloque para reducir contención)
            try:
                self.app_state.add_samples_batch(batch_voltages)
            except Exception:
                # Fallback por compatibilidad
                with self.app_state.data_lock:
                    for v in batch_voltages:
                        self.app_state.voltage_buffer.append(v)
                        self.app_state.time_buffer.append(self.app_state.sample_count)
                        self.app_state.sample_count += 1

            # Mantener tasa de muestreo mediante sleep adaptativo
            elapsed    = time.perf_counter() - t_start
            sleep_time = batch_interval - elapsed
            if sleep_time > 0.001:
                time.sleep(sleep_time)

    # ==============================================================
    # ── PARSER HIBRIDO ASCII / BINARIO ────────────────────────────
    # ==============================================================

    def _push_sample_adc(self, adc: int):
        """Convierte valor ADC de 12 bits a Voltios y lo agrega al buffer."""
        adc_int = int(adc)
        if adc_int < 0 or adc_int > self._adc_max:
            return
        volts = (adc_int * self._vref) / float(self._adc_max)
        self.app_state.add_sample(volts)

    def _handle_text_line(self, line: str):
        """Procesa una linea de texto recibida del ESP32."""
        s = (line or "").strip()
        if not s:
            return

        up = s.upper()

        # Ignorar mensajes de inicio del firmware
        if up in ("READY", "ESP32 ECG READY", "ECG READY", "OK"):
            return

        # ACK de cambio de derivada: "OK:3:aVR"
        if s.startswith("OK:"):
            parts = s.split(":")
            if len(parts) >= 2:
                try:
                    idx = int(parts[1])
                    with self.app_state.mux_lock:
                        self.app_state.current_mux_state = idx
                except Exception:
                    pass
            return

        # Canal dual: "2048,2050" → usar el primer valor
        if "," in s:
            first = s.split(",", 1)[0].strip()
            if first.lstrip("-").isdigit():
                self._push_sample_adc(int(first))
            return

        # Entero simple: muestra ADC
        if s.lstrip("-").isdigit():
            self._push_sample_adc(int(s))
            return

        if getattr(config, "ENABLE_DEBUG_PRINTS", False):
            print(f"[Serial] UART: {s}")

    def _parse_rx_buffer(self):
        """
        Parser hibrido de datos recibidos:
        - Binario  : [0xAA][LSB][MSB][XOR] (4 bytes, verificacion XOR)
        - Texto    : lineas terminadas en '\n'
        """
        while True:
            if not self._rx_buf:
                return

            if self._rx_buf[0] != self.PKT_START:
                # Modo texto: buscar salto de linea
                nl  = self._rx_buf.find(b"\n")
                pkt = self._rx_buf.find(bytes([self.PKT_START]))

                # Si hay un byte de inicio binario antes del newline, alinear
                if pkt != -1 and (nl == -1 or pkt < nl):
                    del self._rx_buf[:pkt]
                    continue

                # Todavia no hay newline: esperar mas datos
                if nl == -1:
                    if len(self._rx_buf) > 4096:
                        self._rx_buf.clear()
                    return

                raw_line = self._rx_buf[:nl]
                del self._rx_buf[:nl + 1]
                try:
                    line = raw_line.decode(errors="ignore").strip()
                except Exception:
                    line = ""
                self._handle_text_line(line)
                continue

            # Paquete binario: necesita exactamente 4 bytes
            if len(self._rx_buf) < 4:
                return

            b0, lsb, msb, xor = (
                self._rx_buf[0], self._rx_buf[1],
                self._rx_buf[2], self._rx_buf[3],
            )
            del self._rx_buf[:4]

            # Verificar integridad del paquete con XOR
            if (b0 ^ lsb ^ msb) != xor:
                continue

            adc = (msb << 8) | lsb
            self._push_sample_adc(adc)

    # ==============================================================
    # ── BUCLE PRINCIPAL DEL HILO ──────────────────────────────────
    # ==============================================================

    def run(self):
        """
        Punto de entrada del hilo.
        Decide entre modo hardware y modo simulacion.
        En hardware: bucle de lectura serial con reconexion automatica.
        """
        # Si no se pudo conectar al inicio, arrancar simulacion directamente
        if self.app_state.simulation_mode:
            print("[Serial] Modo SIMULACION activo (sin hardware).")
            self._simulate_loop()
            return

        # Bucle de lectura hardware con reconexion automatica
        while self.running:
            port_ok = (
                self.serial_port is not None
                and getattr(self.serial_port, "is_open", False)
            )

            if not port_ok:
                self._set_connected(False, simulation=False)
                print(f"[Serial] Reintentando conexion en {self._reconnect_sec}s...")
                time.sleep(self._reconnect_sec)
                self._open_port()

                # Si la reconexion fallo tras el primer intento, pasar a simulacion
                if not self.app_state.esp32_connected:
                    print("[Serial] Hardware no disponible. Activando simulacion.")
                    self._simulate_loop()
                    return
                continue

            try:
                n    = getattr(self.serial_port, "in_waiting", 0)
                data = self.serial_port.read(n if n > 0 else 1)

                if data:
                    self._rx_buf.extend(data)
                    self._parse_rx_buffer()
                    self.app_state.esp32_connected = True

            except (serial.SerialException, OSError, PermissionError) as e:
                print(f"[Serial] Error de puerto: {e}")
                self._close_port()
                self._set_connected(False, simulation=True)
                time.sleep(self._reconnect_sec)

            except Exception as e:
                print(f"[Serial] Error inesperado: {e}")
                self._close_port()
                self._set_connected(False, simulation=True)
                time.sleep(self._reconnect_sec)

    # ==============================================================
    # ── COMANDOS AL ESP32 ─────────────────────────────────────────
    # ==============================================================

    def _send(self, cmd: str):
        """Envia un string de comando al ESP32 (sin bloquear el hilo)."""
        if not (self.serial_port and getattr(self.serial_port, "is_open", False)):
            return
        try:
            self.serial_port.write(cmd.encode())
        except Exception as e:
            print(f"[Serial] Error enviando comando '{cmd.strip()}': {e}")
            self._close_port()

    def send_mux_command(self, state: int):
        """
        Envia comando de cambio de derivada al ESP32.
        Formato segun MUX_COMMAND_FORMAT (default: 'STATE_n\n').
        """
        st  = int(state) & 0x07
        cmd = self._mux_cmd_fmt.format(state=st)
        self._send(cmd)

    def send_pace_command(self, amplitude: float, frequency: float):
        """
        Envia comando de marcapasos al ESP32.
        Formato: 'PACE:{amplitud:.2f},{frecuencia:.1f}\n'
        """
        cmd = f"PACE:{float(amplitude):.2f},{float(frequency):.1f}\n"
        self._send(cmd)

    def stop(self):
        """Detiene el hilo de adquisicion y cierra el puerto serial."""
        self.running = False
        self._close_port()
