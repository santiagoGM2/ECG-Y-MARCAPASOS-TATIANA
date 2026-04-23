"""
appUI.py
Interfaz grafica principal del Monitor ECG + Marcapasos.

Tema visual: ICU Dark — fondo negro, waveform cyan, alertas rojo.
Layout    : Header | Grafico ECG (70%) + Sidebar (30%) | Status Bar

Paneles del sidebar (scrollable):
  1. SIGNOS VITALES        - BPM grande, clasificación de ritmo, QRS detectados, calidad de señal
  2. SELECCIÓN DERIVACIÓN  - 6 botones de derivación + escaneo automático
  3. MARCAPASOS            - Disparo manual, amplitud, frecuencia, vista previa bifásica, auto-estimulación
  4. AJUSTES DE SEÑAL      - Umbrales, ganancia, ventana, Y máx, refresco
  5. CONEXIÓN              - Puerto, baud rate, conectar/desconectar, estado
  6. SIMULACIÓN            - Frec. cardiaca, amplitud, ruido, arritmia, tipo de forma
"""

import tkinter as tk
import threading
import queue
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
import time

from . import config
from .data_model import AppState
from .serial_handler import SerialReader, list_available_ports
from .peak_detection import (
    detect_r_peaks,
    calculate_bpm,
    detect_qrs_complex,
    classify_rhythm,
)

# =========================================================
# -------------------- TEMA ICU DARK ----------------------
# =========================================================

DARK_ICU_THEME = {
    "name":          "ICU Dark",
    "bg":            "#0B1220",
    "panel":         "#111827",
    "border":        "#1F2937",
    "text":          "#E5E7EB",
    "muted":         "#94A3B8",
    "title":         "#F8FAFC",
    "primary":       "#3B82F6",
    "primary_active":"#2563EB",
    "accent":        "#22D3EE",
    "accent_active": "#06B6D4",
    "success":       "#10B981",
    "warning":       "#FBBF24",
    "danger":        "#EF4444",
    "danger_active": "#DC2626",
    "neutral_bg":    "#1B2636",
    "neutral_fg":    "#CBD5E1",
    "info_bg":       "#13233B",
    "info_fg":       "#60A5FA",
    "accent_bg":     "#0F2E2B",
    "accent_fg":     "#2DD4BF",
    "success_bg":    "#0F2A23",
    "success_fg":    "#34D399",
    "warning_bg":    "#3A2A0D",
    "warning_fg":    "#FBBF24",
    "danger_bg":     "#3A1418",
    "danger_fg":     "#F87171",
    "button_text":   "#FFFFFF",
    "plot_bg":       "#0F172A",
    "plot_border":   "#243244",
    "plot_text":     "#E5E7EB",
    "grid":          "#1E2D42",
    "ecg_line":      "#22D3EE",
    "qrs_highlight": "#22D3EE",
    "pace_spike":    "#F59E0B",
    "peak":          "#F87171",
    "baseline":      "#334155",
}

# Derivadas estilo ICU: resaltado activo
_LEAD_NAMES = ["I", "II", "III", "aVR", "aVL", "aVF"]


class ECGApp(tk.Tk):
    """
    Aplicacion principal de monitoreo ECG + Marcapasos.
    Gestiona el estado compartido (AppState), el lector serial/simulacion
    (SerialReader) y toda la interfaz grafica.
    """

    def __init__(self):
        super().__init__()

        self.T = DARK_ICU_THEME  # alias corto para el tema

        self.title("Monitor ECG de Signos Vitales")
        self.geometry("1560x940")
        self.minsize(1280, 780)
        self.configure(bg=self.T["bg"])

        self.is_running = True

        # ── Estado compartido y lector serial ─────────────────────
        self.app_state     = AppState(master=self)

        # Iniciar siempre en simulación al abrir la app.
        # La conexión al hardware se hace únicamente al presionar “Conectar”.
        config.SERIAL_PORT = "NONE_SIM"
        self.serial_reader = SerialReader(self.app_state)

        # ── Estado del control de derivadas ───────────────────────
        self.previous_mux_state    = self.app_state.current_mux_state
        self.last_auto_change_time = time.time()

        # ── Variables UI propias de la app ────────────────────────
        self.refresh_interval_var     = tk.IntVar(value=int(getattr(config, "REFRESH_INTERVAL", 80)))
        self.auto_switch_interval_var = tk.DoubleVar(value=float(getattr(config, "AUTO_SWITCH_INTERVAL", 8.0)))
        self.pace_duration_ms_var     = tk.DoubleVar(value=float(getattr(config, "PACE_SPIKE_DURATION_MS", 4.0)))
        self.pace_alert_hold_var      = tk.DoubleVar(value=float(getattr(config, "PACE_UI_ALERT_SEC", 1.5)))
        self.auto_pacing_var          = tk.BooleanVar(value=False)
        self.auto_scan_active         = False

        # ── Variables de conexion ─────────────────────────────────
        ports              = list_available_ports()
        default_port       = config.SERIAL_PORT if not ports else (
            config.SERIAL_PORT if config.SERIAL_PORT in ports else ports[0]
        )
        self.port_var      = tk.StringVar(value=default_port)
        self.baud_var      = tk.StringVar(value=str(config.BAUDRATE))

        # ── Variables de simulacion ───────────────────────────────
        self.sim_hr_var    = tk.DoubleVar(value=float(getattr(config, "SIMULATION_HEART_RATE", 72)))
        self.sim_amp_var   = tk.DoubleVar(value=1.0)
        self.sim_noise_var = tk.DoubleVar(value=float(getattr(config, "SIMULATION_NOISE", 0.02)))
        self.sim_wf_var    = tk.StringVar(value="ECG NORMAL")

        # ── Estado interno del marcapasos visual ──────────────────
        self._spike_x_sec  = None    # posicion x del spike en el grafico (segundos)
        self._spike_x2_sec = None    # fin del spike (fase negativa)

        # Contador de frames para rate-limiting de paneles (no graficos)
        self._frame_count = 0

        # Indices y caches para evitar trabajo redundante
        self._last_qrs_abs_idx   = 0
        self._last_qrs_complexes = []
        self._vital_cache        = None  # (bpm_round, rhythm, qrs_count, sig_ok)

        # ── Refs a widgets dinamicos ──────────────────────────────
        self._lead_buttons = {}       # {0..5: tk.Button}
        self._qrs_spans    = []       # lista de patches axvspan para QRS

        # ── Construccion de la interfaz ───────────────────────────
        self._create_widgets()

        # ── Actualizaciones iniciales ─────────────────────────────
        self._update_lead_buttons()
        self._update_connection_panel()
        self._update_pacemaker_panel()
        self._update_clock()
        # Dibujar preview bifasico una vez que el canvas tenga tamaño real
        self.after(300, self._draw_biphasic_preview)

        # ── Traces para sincronizar parametros de simulacion ──────
        # Evita actualizar serial_reader cada frame; solo cuando el valor cambia
        for _v in (self.sim_hr_var, self.sim_amp_var, self.sim_noise_var,
                   self.app_state.pace_amplitude_var, self.app_state.pace_bpm_var):
            _v.trace_add("write", self._sync_sim_params)

        # ── Arrancar hilo serial / simulacion ────────────────────
        self.serial_reader.start()

        # ── Programar bucles de actualizacion ────────────────────
        self.after(100, self.update_gui)
        self.after(300, self.check_auto_mode)

        self.protocol("WM_DELETE_WINDOW", self.on_closing)

        # Evitar crashes silenciosos del loop de GUI (Tkinter puede quedar “no responde” si se corta el after)
        self.report_callback_exception = self._on_tk_exception

        # ── Hilo de análisis (picos R / QRS / BPM) para que la GUI no se bloquee ──
        self._analysis_in_q = queue.Queue(maxsize=1)
        self._analysis_out_q = queue.Queue(maxsize=1)
        self._analysis_running = True
        self._analysis_thread = threading.Thread(
            target=self._analysis_loop, name="ECGAnalysisWorker", daemon=True
        )
        self._analysis_thread.start()

        self._analysis_peaks = []
        self._analysis_qrs = []
        self._analysis_bpm = 0.0
        self._analysis_rhythm = "---"

    def _on_tk_exception(self, exc, val, tb):
        """Evita que una excepción rompa el loop `after` y congele la GUI."""
        try:
            import traceback
            traceback.print_exception(exc, val, tb)
        except Exception:
            pass
        try:
            if self.is_running:
                self.after(120, self.update_gui)
        except Exception:
            pass

    def _analysis_loop(self):
        """Hilo de análisis para evitar bloqueos del hilo de GUI."""
        while self._analysis_running:
            job = None
            try:
                job = self._analysis_in_q.get(timeout=0.3)
            except Exception:
                continue
            if not job:
                continue

            try:
                y_centered, sample_rate, r_thr, r_dist = job
                peaks = detect_r_peaks(y_centered, r_thr, r_dist)
                bpm = calculate_bpm(peaks, sample_rate)
                rhythm = classify_rhythm(bpm)
                qrs = detect_qrs_complex(y_centered, peaks, sample_rate)
                result = (peaks, qrs, bpm, rhythm)

                # Dejar solo el resultado más reciente
                try:
                    while True:
                        self._analysis_out_q.get_nowait()
                except Exception:
                    pass
                try:
                    self._analysis_out_q.put_nowait(result)
                except Exception:
                    pass
            except Exception:
                # No matar el hilo por errores puntuales
                continue

    # ==============================================================
    # ── HELPERS SEGUROS ───────────────────────────────────────────
    # ==============================================================

    def _safe_float(self, v, default=0.0):
        try:
            raw = v.get() if hasattr(v, "get") else v
            return float(raw)
        except Exception:
            return float(default)

    def _safe_int(self, v, default=0):
        try:
            raw = v.get() if hasattr(v, "get") else v
            return int(float(raw))
        except Exception:
            return int(default)

    def _sync_sim_params(self, *_):
        """
        Sincroniza parametros de simulacion con SerialReader.
        Se llama via trace cuando cambia cualquier variable de simulacion,
        NO en cada frame del loop principal.
        """
        try:
            self.serial_reader.sim_heart_rate  = self._safe_float(self.sim_hr_var,    72.0)
            self.serial_reader.sim_amplitude   = self._safe_float(self.sim_amp_var,    1.0)
            self.serial_reader.sim_noise_level = self._safe_float(self.sim_noise_var,  0.02)
            self.serial_reader.pace_amplitude  = self._safe_float(self.app_state.pace_amplitude_var, 1.0)
            self.serial_reader.pace_bpm        = self._safe_float(self.app_state.pace_bpm_var,       60.0)
        except Exception:
            pass

    # ==============================================================
    # ── LAYOUT PRINCIPAL ──────────────────────────────────────────
    # ==============================================================

    def _create_widgets(self):
        """Construye la estructura principal: header + body + status bar."""
        root = tk.Frame(self, bg=self.T["bg"])
        root.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Cabecera superior
        self._create_header(root)

        # Cuerpo: grafico ECG + sidebar
        body = tk.Frame(root, bg=self.T["bg"])
        body.pack(fill=tk.BOTH, expand=True, pady=(10, 0))

        # Columna izquierda: panel del grafico (70%)
        ecg_outer = tk.Frame(
            body, bg=self.T["panel"],
            highlightthickness=1, highlightbackground=self.T["border"], bd=0
        )
        ecg_outer.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))
        self._create_ecg_plot(ecg_outer)

        # Columna derecha: sidebar con paneles (30%)
        sidebar_container = tk.Frame(body, bg=self.T["bg"], width=400)
        sidebar_container.pack(side=tk.RIGHT, fill=tk.Y)
        sidebar_container.pack_propagate(False)

        sidebar = self._create_scrollable_sidebar(sidebar_container)

        self._create_vital_signs_panel(sidebar)
        self._create_lead_selection_panel(sidebar)
        self._create_pacemaker_panel(sidebar)
        self._create_signal_settings_panel(sidebar)
        self._create_connection_panel(sidebar)
        self._create_simulation_panel(sidebar)

        # Barra de estado inferior
        self._create_status_bar(root)

    # ----------------------------------------------------------
    def _create_header(self, parent):
        """Barra de cabecera: titulo, badge de conexion, reloj, badge de tema."""
        hdr = tk.Frame(
            parent, bg=self.T["panel"],
            highlightthickness=1, highlightbackground=self.T["border"], bd=0
        )
        hdr.pack(fill=tk.X)

        left = tk.Frame(hdr, bg=self.T["panel"])
        left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=18, pady=14)

        tk.Label(
            left, text="Monitor ECG de Signos Vitales",
            bg=self.T["panel"], fg=self.T["title"],
            font=("Segoe UI", 19, "bold"),
        ).pack(anchor="w")

        tk.Label(
            left, text="Adquisición de 6 derivaciones   |   Análisis ECG en tiempo real   |   Marcapasos bifásico",
            bg=self.T["panel"], fg=self.T["muted"],
            font=("Segoe UI", 9),
        ).pack(anchor="w", pady=(3, 0))

        right = tk.Frame(hdr, bg=self.T["panel"])
        right.pack(side=tk.RIGHT, padx=18, pady=14)

        # Badge de modo (SIMULACIÓN / HARDWARE)
        self.mode_badge_hdr = tk.Label(
            right, text="SIMULACIÓN",
            font=("Segoe UI", 9, "bold"), padx=10, pady=5, bd=0,
        )
        self.mode_badge_hdr.pack(anchor="e", pady=(0, 6))
        self._set_badge(self.mode_badge_hdr, "SIMULACIÓN", "warning")

        # Badge de conexion ESP32
        self.conn_badge_hdr = tk.Label(
            right, text="DESCONECTADO",
            font=("Segoe UI", 9, "bold"), padx=10, pady=5, bd=0,
        )
        self.conn_badge_hdr.pack(anchor="e", pady=(0, 6))
        self._set_badge(self.conn_badge_hdr, "DESCONECTADO", "danger")

        # Reloj
        self.clock_label = tk.Label(
            right, text="--:--:--",
            bg=self.T["panel"], fg=self.T["muted"],
            font=("Segoe UI", 12, "bold"),
        )
        self.clock_label.pack(anchor="e")

    # ----------------------------------------------------------
    def _create_ecg_plot(self, parent):
        """Crea el grafico matplotlib con todos los overlays necesarios."""
        # Titulo del panel ECG
        title_bar = tk.Frame(parent, bg=self.T["panel"])
        title_bar.pack(fill=tk.X, padx=16, pady=(14, 4))

        tk.Label(
            title_bar, text="Señal ECG en Tiempo Real",
            bg=self.T["panel"], fg=self.T["title"],
            font=("Segoe UI", 13, "bold"),
        ).pack(side=tk.LEFT)

        self.lead_title_label = tk.Label(
            title_bar, text="Derivación II",
            bg=self.T["panel"], fg=self.T["accent"],
            font=("Segoe UI", 12, "bold"),
        )
        self.lead_title_label.pack(side=tk.RIGHT)

        divider = tk.Frame(parent, bg=self.T["border"], height=1)
        divider.pack(fill=tk.X, padx=16)

        # ── Figura matplotlib ─────────────────────────────────────
        self.fig, self.ax = plt.subplots(figsize=(10, 6), dpi=98)
        self.fig.patch.set_facecolor(self.T["panel"])
        self.ax.set_facecolor(self.T["plot_bg"])

        for sp in self.ax.spines.values():
            sp.set_color(self.T["plot_border"])

        self.ax.tick_params(colors=self.T["plot_text"], labelsize=9)
        self.ax.set_xlabel("Tiempo (s)", color=self.T["plot_text"], fontsize=10)
        self.ax.set_ylabel("Amplitud (mV)", color=self.T["plot_text"], fontsize=10)
        self.ax.grid(True, color=self.T["grid"], alpha=0.7, linewidth=0.7,
                     linestyle="--", which="both")
        self.ax.margins(x=0)

        # Linea de referencia baseline
        self.baseline_line = self.ax.axhline(
            0, color=self.T["baseline"], linewidth=0.9, alpha=0.7, zorder=1
        )

        # Marcadores de spike de marcapasos: dos lineas verticales (fase+ y fase-)
        # Se usan axvline en lugar de axvspan para evitar manipulacion de poligono
        self._pace_line_pos = self.ax.axvline(
            x=0, color=self.T["pace_spike"], linewidth=2.5,
            linestyle="--", visible=False, zorder=6, alpha=0.95,
            label="Pace +"
        )
        self._pace_line_neg = self.ax.axvline(
            x=0, color=self.T["danger"], linewidth=2.5,
            linestyle="--", visible=False, zorder=6, alpha=0.95,
            label="Pace -"
        )

        # Waveform ECG principal
        self.ecg_line, = self.ax.plot(
            [], [], linewidth=1.7, color=self.T["ecg_line"],
            solid_capstyle="round", zorder=4
        )

        # Linea de resaltado QRS (usa NaN para ocultar regiones fuera del QRS)
        self.qrs_line, = self.ax.plot(
            [], [], linewidth=5, color=self.T["qrs_highlight"],
            alpha=0.35, zorder=3, solid_capstyle="round"
        )

        # Marcadores de picos R
        self.peaks_line, = self.ax.plot(
            [], [], linestyle="", marker="o", markersize=6,
            color=self.T["peak"], zorder=7, markeredgewidth=0
        )

        self.fig.subplots_adjust(left=0.07, right=0.98, top=0.96, bottom=0.10)

        self.mpl_canvas = FigureCanvasTkAgg(self.fig, master=parent)
        mpl_widget = self.mpl_canvas.get_tk_widget()
        mpl_widget.pack(fill=tk.BOTH, expand=True, padx=4, pady=(4, 4))
        mpl_widget.configure(bg=self.T["panel"], highlightthickness=0)

    # ----------------------------------------------------------
    def _create_scrollable_sidebar(self, parent):
        """Retorna el frame interno del sidebar con scroll vertical."""
        outer = tk.Frame(parent, bg=self.T["bg"], bd=0, highlightthickness=0)
        outer.pack(fill=tk.BOTH, expand=True)

        self._sb_canvas = tk.Canvas(
            outer, bg=self.T["bg"], highlightthickness=0, bd=0
        )
        self._sb_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        sb = tk.Scrollbar(outer, orient="vertical", command=self._sb_canvas.yview)
        sb.pack(side=tk.RIGHT, fill=tk.Y)
        self._sb_canvas.configure(yscrollcommand=sb.set)

        self._sb_inner = tk.Frame(self._sb_canvas, bg=self.T["bg"])
        self._sb_win   = self._sb_canvas.create_window(
            (0, 0), window=self._sb_inner, anchor="nw"
        )

        self._sb_inner.bind("<Configure>", lambda _: self._sb_canvas.configure(
            scrollregion=self._sb_canvas.bbox("all")
        ))
        self._sb_canvas.bind("<Configure>", self._on_sb_canvas_resize)

        for widget in (self._sb_canvas, self._sb_inner):
            widget.bind("<Enter>", lambda _: self._sb_canvas.bind_all(
                "<MouseWheel>", self._on_mousewheel
            ))
            widget.bind("<Leave>", lambda _: self._sb_canvas.unbind_all("<MouseWheel>"))

        return self._sb_inner

    def _on_sb_canvas_resize(self, event):
        self._sb_canvas.itemconfig(self._sb_win, width=event.width)

    def _on_mousewheel(self, event):
        self._sb_canvas.yview_scroll(int(-event.delta / 120), "units")

    # ----------------------------------------------------------
    def _create_status_bar(self, parent):
        """Barra de estado en la parte inferior de la ventana."""
        bar = tk.Frame(parent, bg=self.T["panel"],
                       highlightthickness=1, highlightbackground=self.T["border"],
                       height=34)
        bar.pack(fill=tk.X, pady=(8, 0))
        bar.pack_propagate(False)

        def _status_label(text):
            return tk.Label(bar, text=text, bg=self.T["panel"],
                            fg=self.T["muted"], font=("Segoe UI", 9))

        _status_label("  Muestras:").pack(side=tk.LEFT)
        self.sb_samples_lbl = _status_label("0")
        self.sb_samples_lbl.pack(side=tk.LEFT)

        tk.Frame(bar, bg=self.T["border"], width=1).pack(side=tk.LEFT, fill=tk.Y, padx=10, pady=6)

        _status_label("BPM:").pack(side=tk.LEFT)
        self.sb_bpm_lbl = tk.Label(bar, text="---", bg=self.T["panel"],
                                   fg=self.T["success"], font=("Segoe UI", 9, "bold"))
        self.sb_bpm_lbl.pack(side=tk.LEFT, padx=(4, 0))

        tk.Frame(bar, bg=self.T["border"], width=1).pack(side=tk.LEFT, fill=tk.Y, padx=10, pady=6)

        _status_label("Ritmo:").pack(side=tk.LEFT)
        self.sb_rhythm_lbl = tk.Label(bar, text="---", bg=self.T["panel"],
                                      fg=self.T["muted"], font=("Segoe UI", 9, "bold"))
        self.sb_rhythm_lbl.pack(side=tk.LEFT, padx=(4, 0))

        tk.Frame(bar, bg=self.T["border"], width=1).pack(side=tk.LEFT, fill=tk.Y, padx=10, pady=6)

        self.sb_mode_lbl = tk.Label(bar, text="MODO SIMULACIÓN", bg=self.T["panel"],
                                    fg=self.T["warning"], font=("Segoe UI", 9, "bold"))
        self.sb_mode_lbl.pack(side=tk.RIGHT, padx=12)

        _status_label("  Frec.: 1000 Hz  |  Sesión: DEMO BIOMÉDICA  |").pack(side=tk.RIGHT)

    # ==============================================================
    # ── HELPERS DE WIDGETS ────────────────────────────────────────
    # ==============================================================

    def _create_panel(self, parent, title, subtitle=None, color_bar=None):
        """Crea un panel con cabecera, separador y frame de contenido."""
        outer = tk.Frame(
            parent, bg=self.T["panel"],
            highlightthickness=1, highlightbackground=self.T["border"], bd=0
        )
        outer.pack(fill=tk.X, pady=(0, 10))

        # Barra de color izquierda (opcional)
        if color_bar:
            accent_bar = tk.Frame(outer, bg=color_bar, width=3)
            accent_bar.pack(side=tk.LEFT, fill=tk.Y)

        inner = tk.Frame(outer, bg=self.T["panel"])
        inner.pack(fill=tk.BOTH, expand=True)

        hdr = tk.Frame(inner, bg=self.T["panel"])
        hdr.pack(fill=tk.X, padx=14, pady=(12, 6))

        tk.Label(
            hdr, text=title,
            bg=self.T["panel"], fg=self.T["title"],
            font=("Segoe UI", 11, "bold"),
        ).pack(anchor="w")

        if subtitle:
            tk.Label(
                hdr, text=subtitle,
                bg=self.T["panel"], fg=self.T["muted"],
                font=("Segoe UI", 8),
            ).pack(anchor="w", pady=(2, 0))

        tk.Frame(inner, bg=self.T["border"], height=1).pack(fill=tk.X, padx=14)

        content = tk.Frame(inner, bg=self.T["panel"])
        content.pack(fill=tk.BOTH, expand=True, padx=14, pady=(10, 12))

        outer.content = content
        return outer

    def _set_badge(self, widget, text, kind="neutral", font=None):
        """Aplica color de badge a un Label segun el tipo (success/warning/danger...)."""
        palettes = {
            "neutral": (self.T["neutral_bg"], self.T["neutral_fg"]),
            "info":    (self.T["info_bg"],    self.T["info_fg"]),
            "accent":  (self.T["accent_bg"],  self.T["accent_fg"]),
            "success": (self.T["success_bg"], self.T["success_fg"]),
            "warning": (self.T["warning_bg"], self.T["warning_fg"]),
            "danger":  (self.T["danger_bg"],  self.T["danger_fg"]),
        }
        bg, fg = palettes.get(kind, palettes["neutral"])
        widget.config(text=text, bg=bg, fg=fg,
                      font=font or ("Segoe UI", 9, "bold"))

    def _btn(self, parent, text, command, kind="primary", font=None, padx=12, pady=9):
        """Boton de accion con estilo consistente."""
        colors = {
            "primary": (self.T["primary"],  self.T["primary_active"]),
            "accent":  (self.T["accent"],   self.T["accent_active"]),
            "danger":  (self.T["danger"],   self.T["danger_active"]),
            "success": (self.T["success"],  "#0d9268"),
            "neutral": (self.T["neutral_bg"], "#253649"),
        }
        bg, abg = colors.get(kind, colors["primary"])
        return tk.Button(
            parent, text=text, command=command,
            bg=bg, fg=self.T["button_text"],
            activebackground=abg, activeforeground=self.T["button_text"],
            relief="flat", bd=0, cursor="hand2",
            font=font or ("Segoe UI", 9, "bold"),
            padx=padx, pady=pady, highlightthickness=0,
        )

    def _spinbox(self, parent, var, from_, to, increment, width=10):
        """Spinbox numerica con estilo ICU Dark."""
        sb = tk.Spinbox(
            parent, from_=from_, to=to, increment=increment,
            textvariable=var, width=width,
            font=("Segoe UI", 9), justify="center",
            bg=self.T["neutral_bg"], fg=self.T["text"],
            relief="flat", bd=0,
            buttonbackground=self.T["panel"],
            highlightthickness=1, highlightbackground=self.T["border"],
            highlightcolor=self.T["primary"],
            insertbackground=self.T["text"],
        )
        return sb

    def _row(self, parent, label_text, help_text=None):
        """Fila de dos columnas: etiqueta (izquierda) + widget (derecha)."""
        row = tk.Frame(parent, bg=self.T["panel"])
        row.pack(fill=tk.X, pady=4)

        left = tk.Frame(row, bg=self.T["panel"])
        left.pack(side=tk.LEFT, fill=tk.X, expand=True)

        tk.Label(left, text=label_text,
                 bg=self.T["panel"], fg=self.T["text"],
                 font=("Segoe UI", 9, "bold")).pack(anchor="w")

        if help_text:
            tk.Label(left, text=help_text,
                     bg=self.T["panel"], fg=self.T["muted"],
                     font=("Segoe UI", 7), wraplength=160,
                     justify="left").pack(anchor="w")

        right = tk.Frame(row, bg=self.T["panel"])
        right.pack(side=tk.RIGHT, anchor="e")
        return right

    def _metric_row(self, parent, label_text):
        """Fila de metrica: etiqueta + badge de valor."""
        row = tk.Frame(parent, bg=self.T["panel"])
        row.pack(fill=tk.X, pady=3)

        tk.Label(row, text=label_text,
                 bg=self.T["panel"], fg=self.T["text"],
                 font=("Segoe UI", 9)).pack(side=tk.LEFT)

        badge = tk.Label(row, text="---", padx=8, pady=4, bd=0)
        badge.pack(side=tk.RIGHT)
        self._set_badge(badge, "---", "neutral")
        return badge

    # ==============================================================
    # ── PANEL 1: SIGNOS VITALES ───────────────────────────────────
    # ==============================================================

    def _create_vital_signs_panel(self, parent):
        p = self._create_panel(parent, "Signos Vitales",
                               "Parámetros cardiacos en tiempo real", color_bar=self.T["success"])
        c = p.content

        # BPM grande
        self.bpm_big_label = tk.Label(
            c, text="---",
            bg=self.T["panel"], fg=self.T["success"],
            font=("Segoe UI", 36, "bold"),
        )
        self.bpm_big_label.pack(anchor="center", pady=(4, 0))

        tk.Label(c, text="latidos por minuto",
                 bg=self.T["panel"], fg=self.T["muted"],
                 font=("Segoe UI", 8)).pack(anchor="center")

        # Badge de clasificacion del ritmo
        self.rhythm_badge = tk.Label(
            c, text="ASISTOLIA", padx=14, pady=6, bd=0,
            font=("Segoe UI", 11, "bold")
        )
        self.rhythm_badge.pack(fill=tk.X, pady=(8, 4))
        self._set_badge(self.rhythm_badge, "ASISTOLIA", "neutral",
                        font=("Segoe UI", 11, "bold"))

        # Metricas adicionales
        self.qrs_count_badge  = self._metric_row(c, "QRS detectados")
        self.sig_quality_badge = self._metric_row(c, "Calidad de señal")

    # ==============================================================
    # ── PANEL 2: SELECCIÓN DE DERIVACIÓN ──────────────────────────
    # ==============================================================

    def _create_lead_selection_panel(self, parent):
        p = self._create_panel(parent, "Selección de Derivación",
                               "Derivación de ECG (MUX)", color_bar=self.T["primary"])
        c = p.content

        # Indicador de derivada activa
        self.active_lead_badge = tk.Label(
            c, text="DERIVACIÓN II", padx=10, pady=6, bd=0,
            font=("Segoe UI", 12, "bold")
        )
        self.active_lead_badge.pack(fill=tk.X, pady=(0, 8))
        self._set_badge(self.active_lead_badge, "DERIVACIÓN II", "info",
                        font=("Segoe UI", 12, "bold"))

        # Grid 2x3 de botones de derivada
        grid = tk.Frame(c, bg=self.T["panel"])
        grid.pack(fill=tk.X, pady=(0, 8))

        leads = list(zip(_LEAD_NAMES, range(6)))
        for idx, (name, state) in enumerate(leads):
            btn = self._btn(grid, name, lambda s=state: self.on_lead_select(s),
                            kind="neutral", padx=6, pady=7,
                            font=("Segoe UI", 10, "bold"))
            btn.grid(row=idx // 3, column=idx % 3, padx=3, pady=3, sticky="ew")
            self._lead_buttons[state] = btn

        grid.columnconfigure(0, weight=1)
        grid.columnconfigure(1, weight=1)
        grid.columnconfigure(2, weight=1)

        # AUTO SCAN toggle
        self.auto_scan_btn = self._btn(
            c, "ESCANEO AUTO  APAGADO", self.on_auto_scan_toggle,
            kind="neutral", pady=7
        )
        self.auto_scan_btn.pack(fill=tk.X, pady=(4, 0))

        # Intervalo de auto-scan
        row = self._row(c, "Intervalo automático (s)", "Segundos entre cambios de derivación")
        self._spinbox(row, self.auto_switch_interval_var, 1.0, 30.0, 1.0, 8).pack()

    # ==============================================================
    # ── PANEL 3: PACEMAKER CONTROL ────────────────────────────────
    # ==============================================================

    def _create_pacemaker_panel(self, parent):
        p = self._create_panel(parent, "Control de Marcapasos",
                               "Estimulación con pulso bifásico", color_bar=self.T["danger"])
        c = p.content

        # Badge de estado del marcapasos
        self.pace_status_badge = tk.Label(
            c, text="SIN ALERTA", padx=12, pady=8, bd=0,
            font=("Segoe UI", 12, "bold")
        )
        self.pace_status_badge.pack(fill=tk.X, pady=(0, 6))
        self._set_badge(self.pace_status_badge, "SIN ALERTA", "success",
                        font=("Segoe UI", 12, "bold"))

        # BOTON TRIGGER (prominente)
        self._btn(c, "DISPARAR PULSO", self.on_pace_trigger,
                  kind="danger", pady=12,
                  font=("Segoe UI", 12, "bold")).pack(fill=tk.X, pady=(0, 10))

        # Controles de parametros del marcapasos
        r = self._row(c, "Amplitud (V)", "Altura del pulso (0.1 - 3.0 V)")
        self._spinbox(r, self.app_state.pace_amplitude_var, 0.1, 3.0, 0.1, 8).pack()

        r = self._row(c, "Frecuencia (BPM)", "Tasa de estimulación (30 - 200 BPM)")
        self._spinbox(r, self.app_state.pace_bpm_var, 30.0, 200.0, 1.0, 8).pack()

        r = self._row(c, "Duración (ms)", "Ancho del pulso bifásico (1 - 30 ms)")
        self._spinbox(r, self.pace_duration_ms_var, 1.0, 30.0, 1.0, 8).pack()

        # Preview del pulso bifasico
        tk.Label(c, text="Vista previa de la forma de onda bifásica:",
                 bg=self.T["panel"], fg=self.T["muted"],
                 font=("Segoe UI", 8)).pack(anchor="w", pady=(8, 2))

        preview_frame = tk.Frame(c, bg=self.T["plot_bg"],
                                 highlightthickness=1, highlightbackground=self.T["border"],
                                 height=70)
        preview_frame.pack(fill=tk.X)
        preview_frame.pack_propagate(False)

        self.pace_canvas = tk.Canvas(preview_frame, bg=self.T["plot_bg"],
                                     highlightthickness=0, height=70)
        self.pace_canvas.pack(fill=tk.BOTH, expand=True)
        # Redibujar preview solo cuando cambia el tamaño del canvas (no cada frame)
        self.pace_canvas.bind("<Configure>", lambda _: self.after_idle(self._draw_biphasic_preview))

        # Checkbox auto-pacing
        self.auto_pacing_var.trace_add("write", self._on_auto_pacing_change)
        chk = tk.Checkbutton(
            c, text="Habilitar auto-estimulación",
            variable=self.auto_pacing_var,
            bg=self.T["panel"], fg=self.T["text"],
            selectcolor=self.T["neutral_bg"],
            activebackground=self.T["panel"],
            activeforeground=self.T["text"],
            font=("Segoe UI", 9),
        )
        chk.pack(anchor="w", pady=(8, 0))

        # Fila de metricas del marcapasos
        self.pace_amp_badge  = self._metric_row(c, "Amplitud del pulso")
        self.pace_dur_badge  = self._metric_row(c, "Duración del pulso")
        self.pace_bpm_badge  = self._metric_row(c, "Frecuencia de estimulación")

    def _draw_biphasic_preview(self):
        """Dibuja la forma de onda bifasica en el canvas de preview."""
        canvas = self.pace_canvas
        canvas.update_idletasks()
        W = canvas.winfo_width() or 360
        H = 70

        canvas.delete("all")

        # Colores de la forma de onda bifasica
        wave_pos = "#F59E0B"   # naranja — fase positiva
        wave_neg = "#F87171"   # rojo suave — fase negativa
        base_col = self.T["baseline"]

        cy  = H // 2            # eje Y central
        amp = int(H * 0.34)     # amplitud en pixeles
        m   = int(W * 0.08)     # margen lateral

        # Coordenadas X de cada segmento del waveform
        x0 = m
        x1 = m + int((W - 2*m) * 0.25)
        x2 = m + int((W - 2*m) * 0.50)
        x3 = m + int((W - 2*m) * 0.75)
        x4 = W - m

        # Linea de baseline (punteada)
        canvas.create_line(0, cy, W, cy, fill=base_col, width=1, dash=(3, 3))

        # Etiquetas de fase
        canvas.create_text(int((x1 + x2) / 2), cy - amp - 8,
                           text="+", fill=wave_pos, font=("Segoe UI", 9, "bold"))
        canvas.create_text(int((x2 + x3) / 2), cy + amp + 8,
                           text="-", fill=wave_neg, font=("Segoe UI", 9, "bold"))

        # Segmento positivo: baseline → subida → fase alta → borde
        canvas.create_line(
            x0, cy, x1, cy, x1, cy - amp, x2, cy - amp,
            fill=wave_pos, width=2, smooth=False
        )
        # Segmento negativo: borde → fase baja → bajada → baseline
        canvas.create_line(
            x2, cy + amp, x3, cy + amp, x3, cy, x4, cy,
            fill=wave_neg, width=2, smooth=False
        )
        # Transicion vertical instantanea entre fases
        canvas.create_line(x2, cy - amp, x2, cy + amp, fill=self.T["muted"], width=1)

        # Etiquetas de tiempo
        canvas.create_text(x0, H - 4, text="0", fill=self.T["muted"], font=("Segoe UI", 7))
        canvas.create_text(x4, H - 4, text="T", fill=self.T["muted"], font=("Segoe UI", 7))

    # ==============================================================
    # ── PANEL 4: AJUSTES DE SEÑAL ─────────────────────────────────
    # ==============================================================

    def _create_signal_settings_panel(self, parent):
        p = self._create_panel(parent, "Ajustes de Señal",
                               "Umbrales de detección y visualización", color_bar=self.T["accent"])
        c = p.content

        self.acq_rate_badge = self._metric_row(c, "Frecuencia de muestreo")
        self._set_badge(self.acq_rate_badge, f"{getattr(config,'SAMPLE_RATE',1000)} Hz", "info")

        params = [
            ("Umbral R (V)",  "Voltaje mínimo de pico para detectar R",
             self.app_state.r_threshold,   0.05, 3.0,  0.05),
            ("Distancia R (muestras)","Mínimo de muestras entre picos R",
             self.app_state.r_distance,    50,   600,  10),
            ("Ganancia ECG",         "Amplificación vertical de la señal",
             self.app_state.ecg_gain,      0.1,  10.0, 0.1),
            ("Ventana (muestras)", "Número de muestras visibles en el gráfico",
             self.app_state.window_size,   200,  5000, 100),
            ("Y máx (V)",        "Límite del eje vertical",
             self.app_state.y_max,         0.5,  10.0, 0.1),
            ("Refresco (ms)",     "Intervalo de actualización de la GUI",
             self.refresh_interval_var,    20,   500,  10),
        ]

        for lbl, help_txt, var, fr, to, inc in params:
            r = self._row(c, lbl, help_txt)
            self._spinbox(r, var, fr, to, inc, 9).pack()

    # ==============================================================
    # ── PANEL 5: CONEXIÓN ─────────────────────────────────────────
    # ==============================================================

    def _create_connection_panel(self, parent):
        p = self._create_panel(parent, "Conexión",
                               "Puerto serial y estado del hardware", color_bar=self.T["warning"])
        c = p.content

        # Badge de modo grande
        self.hw_mode_badge = tk.Label(
            c, text="MODO SIMULACIÓN", padx=12, pady=8, bd=0,
            font=("Segoe UI", 11, "bold")
        )
        self.hw_mode_badge.pack(fill=tk.X, pady=(0, 8))
        self._set_badge(self.hw_mode_badge, "MODO SIMULACIÓN", "warning",
                        font=("Segoe UI", 11, "bold"))

        # Selector de puerto COM
        r_port = self._row(c, "Puerto COM", "Selecciona el puerto serial")
        self._available_ports = list_available_ports() or [config.SERIAL_PORT]
        self._port_menu = tk.OptionMenu(r_port, self.port_var, *self._available_ports)
        self._port_menu.configure(
            bg=self.T["neutral_bg"], fg=self.T["text"],
            activebackground=self.T["primary"], activeforeground="white",
            highlightthickness=0, relief="flat", font=("Segoe UI", 9),
            width=10,
        )
        self._port_menu["menu"].configure(
            bg=self.T["neutral_bg"], fg=self.T["text"],
            activebackground=self.T["primary"], activeforeground="white",
        )
        self._port_menu.pack()

        # Baud rate
        r_baud = self._row(c, "Baud rate", "Debe coincidir con el firmware del ESP32")
        baud_options = ["9600", "19200", "57600", "115200", "230400"]
        baud_menu = tk.OptionMenu(r_baud, self.baud_var, *baud_options)
        baud_menu.configure(
            bg=self.T["neutral_bg"], fg=self.T["text"],
            activebackground=self.T["primary"], activeforeground="white",
            highlightthickness=0, relief="flat", font=("Segoe UI", 9),
            width=10,
        )
        baud_menu["menu"].configure(
            bg=self.T["neutral_bg"], fg=self.T["text"],
            activebackground=self.T["primary"], activeforeground="white",
        )
        baud_menu.pack()

        # Botones de conexion
        btn_row = tk.Frame(c, bg=self.T["panel"])
        btn_row.pack(fill=tk.X, pady=(8, 0))

        self.connect_btn = self._btn(
            btn_row, "Conectar", self.on_connect, kind="primary", pady=8
        )
        self.connect_btn.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 4))

        self._btn(btn_row, "Actualizar puertos", self.on_refresh_ports,
                  kind="neutral", pady=8).pack(side=tk.RIGHT)

        # Metricas
        self.conn_esp32_badge  = self._metric_row(c, "Estado del ESP32")
        self.conn_samples_badge = self._metric_row(c, "Muestras totales")

    # ==============================================================
    # ── PANEL 6: CONTROL DE SIMULACIÓN ────────────────────────────
    # ==============================================================

    def _create_simulation_panel(self, parent):
        p = self._create_panel(parent, "Control de Simulación",
                               "Parámetros del generador de ECG", color_bar=self.T["accent"])
        c = p.content

        params = [
            ("Frecuencia cardiaca (BPM)", "Frecuencia simulada (30-200)",
             self.sim_hr_var,    30.0, 200.0, 1.0),
            ("Amplitud del ECG",    "Multiplicador de amplitud de la onda R",
             self.sim_amp_var,   0.1,  3.0,   0.1),
            ("Nivel de ruido (mV)", "Desviación estándar del ruido gaussiano",
             self.sim_noise_var, 0.0,  0.5,   0.01),
        ]

        for lbl, help_txt, var, fr, to, inc in params:
            r = self._row(c, lbl, help_txt)
            self._spinbox(r, var, fr, to, inc, 9).pack()

        # Selector de tipo de forma de onda
        r_wf = self._row(c, "Tipo de forma de onda", "Condición cardiaca predefinida")
        wf_options = ["ECG NORMAL", "BRADICARDIA", "TAQUICARDIA"]
        wf_menu = tk.OptionMenu(r_wf, self.sim_wf_var, *wf_options,
                                command=self._on_waveform_type_change)
        wf_menu.configure(
            bg=self.T["neutral_bg"], fg=self.T["text"],
            activebackground=self.T["primary"], activeforeground="white",
            highlightthickness=0, relief="flat", font=("Segoe UI", 9), width=14
        )
        wf_menu["menu"].configure(
            bg=self.T["neutral_bg"], fg=self.T["text"],
            activebackground=self.T["primary"], activeforeground="white",
        )
        wf_menu.pack()

        # Boton de arritmia
        self._btn(c, "Agregar arritmia (5 s)", self.on_add_arrhythmia,
                  kind="warning", pady=8).pack(fill=tk.X, pady=(10, 0))

        self.sim_status_badge = self._metric_row(c, "Estado del generador")
        self._set_badge(self.sim_status_badge, "EN EJECUCIÓN", "success")

    # ==============================================================
    # ── CLOCK ─────────────────────────────────────────────────────
    # ==============================================================

    def _update_clock(self):
        if not self.is_running:
            return
        self.clock_label.config(text=time.strftime("%H:%M:%S"))
        self.after(1000, self._update_clock)

    # ==============================================================
    # ── ECG PLOT HELPERS ──────────────────────────────────────────
    # ==============================================================

    def _signal_present(self, y_np: np.ndarray) -> bool:
        """Determina si la señal tiene amplitud suficiente para ser valida."""
        if y_np is None or len(y_np) < 20:
            return False
        win  = min(len(y_np), getattr(config, "NO_SIGNAL_WINDOW_SAMPLES", 500))
        seg  = y_np[-win:]
        p2p  = float(np.max(seg) - np.min(seg))
        std  = float(np.std(seg))
        thr  = float(getattr(config, "NO_SIGNAL_P2P_V", 0.02))
        thr_s = float(getattr(config, "NO_SIGNAL_STD_V", 0.005))
        return p2p >= thr or std >= thr_s

    def _update_no_signal_state(self, present: bool):
        """Actualiza la bandera no_signal con histeresis temporal."""
        now     = time.time()
        min_sec = float(getattr(config, "NO_SIGNAL_TIMEOUT_SEC", 1.0))
        if present:
            self.app_state.no_signal       = False
            self.app_state.no_signal_since = None
            return
        if self.app_state.no_signal_since is None:
            self.app_state.no_signal_since = now
        if (now - self.app_state.no_signal_since) >= min_sec:
            self.app_state.no_signal = True

    def _inject_biphasic_spike(self, y_signal, duration_ms, amplitude):
        """
        Inyecta un pulso bifasico al final de la señal visualizada.

        Fase positiva (+amplitude) durante la primera mitad,
        fase negativa (-amplitude) durante la segunda mitad.

        Retorna:
            (y_out, x_start_idx, x_end_idx) — señal modificada e indices del spike
        """
        y_out       = np.array(y_signal, dtype=float).copy()
        sample_rate = int(getattr(config, "SAMPLE_RATE", 1000))
        total_s     = max(2, int(round(sample_rate * duration_ms / 1000.0)))
        half_s      = total_s // 2

        if len(y_out) < total_s + 4:
            return y_out, 0, 0

        end_idx   = len(y_out) - 2
        start_idx = max(0, end_idx - total_s)

        # Baseline local: mediana de los 50ms previos
        ref_start    = max(0, start_idx - int(sample_rate * 0.05))
        local_base   = float(np.median(y_out[ref_start:start_idx])) if start_idx > ref_start else 0.0

        # Fase positiva
        y_out[start_idx:start_idx + half_s] = local_base + amplitude
        # Fase negativa
        y_out[start_idx + half_s:end_idx]   = local_base - amplitude
        # Transiciones nitidas
        if start_idx > 0:
            y_out[start_idx - 1] = local_base
        if end_idx < len(y_out):
            y_out[end_idx] = local_base

        return y_out, start_idx, end_idx

    # ==============================================================
    # ── ACTUALIZADORES DE PANELES ─────────────────────────────────
    # ==============================================================

    def _update_vital_signs(self, bpm: float, rhythm: str, qrs_count: int, sig_ok: bool):
        """Actualiza el panel SIGNOS VITALES con los valores del frame actual."""
        # Color del BPM segun el ritmo
        if bpm <= 0:
            bpm_text  = "---"
            bpm_color = self.T["muted"]
        elif bpm < 40 or bpm > 120:
            bpm_text  = f"{bpm:.0f}"
            bpm_color = self.T["danger"]
        elif bpm < 60 or bpm > 100:
            bpm_text  = f"{bpm:.0f}"
            bpm_color = self.T["warning"]
        else:
            bpm_text  = f"{bpm:.0f}"
            bpm_color = self.T["success"]

        self.bpm_big_label.config(text=bpm_text, fg=bpm_color)

        # Badge de ritmo
        rhythm_upper = (rhythm or "").upper()
        rhythm_ui_map = {
            "NORMAL": "NORMAL",
            "BRADYCARDIA": "BRADICARDIA",
            "TACHYCARDIA": "TAQUICARDIA",
            "ASYSTOLE": "ASISTOLIA",
        }
        rhythm_colors = {
            "NORMAL": "success",
            "BRADYCARDIA": "warning",
            "TACHYCARDIA": "warning",
            "ASYSTOLE": "danger",
        }
        kind = rhythm_colors.get(rhythm_upper, "neutral")
        self._set_badge(self.rhythm_badge, rhythm_ui_map.get(rhythm_upper, rhythm_upper or "---"), kind,
                        font=("Segoe UI", 11, "bold"))

        # Conteo QRS
        self._set_badge(self.qrs_count_badge, str(qrs_count), "info")

        # Calidad de señal
        if not sig_ok:
            self._set_badge(self.sig_quality_badge, "SIN SEÑAL", "neutral")
        elif bpm > 0:
            self._set_badge(self.sig_quality_badge, "BUENA", "success")
        else:
            self._set_badge(self.sig_quality_badge, "BAJA", "warning")

    def _update_lead_buttons(self):
        """Resalta el boton de la derivada activa."""
        active = self.app_state.current_mux_state
        label  = self.app_state.mux_state_label.get(active, "?")

        for state, btn in self._lead_buttons.items():
            if state == active:
                btn.config(bg=self.T["primary"], fg=self.T["button_text"])
            else:
                btn.config(bg=self.T["neutral_bg"], fg=self.T["text"])

        self._set_badge(self.active_lead_badge, f"DERIVACIÓN {label}", "info",
                        font=("Segoe UI", 12, "bold"))

        if hasattr(self, "lead_title_label"):
            self.lead_title_label.config(text=f"Derivación {label}")

        if hasattr(self, "ax"):
            self.ax.set_title(f"Señal ECG  —  Derivación {label}",
                              color=self.T["plot_text"], fontsize=11, pad=6)

    def _update_pacemaker_panel(self):
        """Actualiza el badge de estado y los chips de parametros del marcapasos."""
        now = time.time()

        if now < getattr(self.app_state, "pace_alert_until", 0.0):
            kind = "danger"
            text = "MARCAPASOS ACTIVO"
        elif getattr(self.app_state, "no_signal", False):
            kind = "neutral"
            text = "SIN SEÑAL"
        else:
            kind = "success"
            text = "SIN ALERTA"

        self._set_badge(self.pace_status_badge, text, kind,
                        font=("Segoe UI", 12, "bold"))

        amp = self._safe_float(self.app_state.pace_amplitude_var, 1.0)
        dur = self._safe_float(self.pace_duration_ms_var, 4.0)
        bpm = self._safe_float(self.app_state.pace_bpm_var, 60.0)

        self._set_badge(self.pace_amp_badge,  f"{amp:.1f} V",   "danger")
        self._set_badge(self.pace_dur_badge,  f"{dur:.0f} ms",  "info")
        self._set_badge(self.pace_bpm_badge,  f"{bpm:.0f} BPM", "accent")

    def _update_connection_panel(self):
        """Actualiza badges de conexion segun el estado actual."""
        sim  = getattr(self.app_state, "simulation_mode", True)
        conn = self.app_state.esp32_connected

        if conn:
            self._set_badge(self.hw_mode_badge,   "MODO HARDWARE",  "success",
                            font=("Segoe UI", 11, "bold"))
            self._set_badge(self.conn_esp32_badge, "EN LÍNEA",        "success")
            self.connect_btn.config(text="Desconectar", bg=self.T["danger"])
            self._set_badge(self.conn_badge_hdr,  "EN LÍNEA",  "success")
            self._set_badge(self.mode_badge_hdr,  "HARDWARE", "success")
        else:
            if sim:
                self._set_badge(self.hw_mode_badge,   "MODO SIMULACIÓN", "warning",
                                font=("Segoe UI", 11, "bold"))
                self._set_badge(self.conn_badge_hdr,  "SIMULACIÓN",      "warning")
                self._set_badge(self.mode_badge_hdr,  "SIMULACIÓN",      "warning")
            else:
                self._set_badge(self.hw_mode_badge,   "DESCONECTADO",   "danger",
                                font=("Segoe UI", 11, "bold"))
                self._set_badge(self.conn_badge_hdr,  "DESCONECTADO",   "danger")
                self._set_badge(self.mode_badge_hdr,  "DESCONECTADO",   "danger")

            self._set_badge(self.conn_esp32_badge, "DESCONECTADO", "danger")
            self.connect_btn.config(text="Conectar", bg=self.T["primary"])

        self._set_badge(self.conn_samples_badge,
                        f"{self.app_state.sample_count:,}", "neutral")

        # Status bar
        mode_text  = "HARDWARE" if conn else "SIMULACIÓN"
        mode_color = self.T["success"] if conn else self.T["warning"]
        self.sb_mode_lbl.config(text=f"MODO {mode_text}", fg=mode_color)

    def _update_simulation_panel(self):
        """Sincroniza los parametros de simulacion con el SerialReader."""
        hr  = self._safe_float(self.sim_hr_var,    72.0)
        amp = self._safe_float(self.sim_amp_var,    1.0)
        nse = self._safe_float(self.sim_noise_var,  0.02)

        self.serial_reader.sim_heart_rate  = hr
        self.serial_reader.sim_amplitude   = amp
        self.serial_reader.sim_noise_level = nse
        self.serial_reader.pace_amplitude  = self._safe_float(
            self.app_state.pace_amplitude_var, 1.0
        )
        self.serial_reader.pace_bpm        = self._safe_float(
            self.app_state.pace_bpm_var, 60.0
        )

        arrhythmia = getattr(self.serial_reader, "sim_arrhythmia", False)
        if arrhythmia:
            self._set_badge(self.sim_status_badge, "ARRITMIA", "warning")
        else:
            self._set_badge(self.sim_status_badge, "EN EJECUCIÓN", "success")

    # ==============================================================
    # ── ACCION: SELECCION DE DERIVADA ────────────────────────────
    # ==============================================================

    def on_lead_select(self, state: int):
        """Selecciona una derivada manualmente y reinicia buffers."""
        prev = self.app_state.current_mux_state
        self.app_state.set_mux_state(state)
        self._send_mux_if_changed(prev)
        self._update_lead_buttons()

    def on_auto_scan_toggle(self):
        """Alterna el modo AUTO SCAN de derivadas."""
        self.auto_scan_active = not self.auto_scan_active
        if self.auto_scan_active:
            self.app_state.operation_mode.set(config.MODE_AUTO)
            self.auto_scan_btn.config(text="ESCANEO AUTO  ENCENDIDO",
                                      bg=self.T["accent"])
        else:
            self.app_state.operation_mode.set(config.MODE_MANUAL)
            self.app_state.last_manual_action_time = time.time()
            self.auto_scan_btn.config(text="ESCANEO AUTO  APAGADO",
                                      bg=self.T["neutral_bg"])

    # ==============================================================
    # ── ACCION: MARCAPASOS ────────────────────────────────────────
    # ==============================================================

    def on_pace_trigger(self):
        """Activa el trigger manual del marcapasos."""
        now = time.time()
        self.app_state.pace_pulse_pending = True
        hold = max(0.5, self._safe_float(self.pace_alert_hold_var, 1.5))
        self.app_state.pace_alert_until = now + hold

        # Enviar comando al ESP32 si esta conectado
        if self.app_state.esp32_connected:
            amp  = self._safe_float(self.app_state.pace_amplitude_var, 1.0)
            freq = self._safe_float(self.app_state.pace_bpm_var, 60.0)
            self.serial_reader.send_pace_command(amp, freq)

    def _on_auto_pacing_change(self, *_):
        """Callback cuando cambia el estado del checkbox de auto-pacing."""
        enabled = self.auto_pacing_var.get()
        self.serial_reader.auto_pacing_enabled = enabled

        if enabled and self.app_state.esp32_connected:
            amp  = self._safe_float(self.app_state.pace_amplitude_var, 1.0)
            freq = self._safe_float(self.app_state.pace_bpm_var, 60.0)
            self.serial_reader.send_pace_command(amp, freq)

    # ==============================================================
    # ── ACCION: CONEXION ─────────────────────────────────────────
    # ==============================================================

    def on_connect(self):
        """Conecta al puerto seleccionado o desconecta el hardware."""
        if self.app_state.esp32_connected:
            # Desconectar → pasar a simulacion
            self._restart_reader(port="NONE_DISCONNECT")
        else:
            # Intentar conectar al puerto seleccionado
            port = self.port_var.get().strip()
            try:
                baud = int(self.baud_var.get())
            except Exception:
                baud = config.BAUDRATE
            config.SERIAL_PORT = port
            config.BAUDRATE    = baud
            self._restart_reader(port=port)

    def _restart_reader(self, port: str):
        """Para el reader actual y arranca uno nuevo."""
        try:
            self.serial_reader.stop()
        except Exception:
            pass

        config.SERIAL_PORT = port

        with self.app_state.data_lock:
            self.app_state.voltage_buffer.clear()
            self.app_state.time_buffer.clear()

        self.app_state.sample_count = 0
        self.serial_reader = SerialReader(self.app_state)
        self.serial_reader.auto_pacing_enabled = self.auto_pacing_var.get()
        self.serial_reader.start()

        self.after(600, self._update_connection_panel)

    def on_refresh_ports(self):
        """Actualiza la lista de puertos COM disponibles."""
        ports = list_available_ports()
        if not ports:
            ports = [config.SERIAL_PORT]
        self._available_ports = ports

        menu = self._port_menu["menu"]
        menu.delete(0, "end")
        for p in ports:
            menu.add_command(label=p, command=lambda v=p: self.port_var.set(v))
        if self.port_var.get() not in ports and ports:
            self.port_var.set(ports[0])

    # ==============================================================
    # ── ACCION: SIMULACION ────────────────────────────────────────
    # ==============================================================

    def on_add_arrhythmia(self):
        """Activa arritmia simulada durante 5 segundos."""
        self.serial_reader.sim_arrhythmia       = True
        self.serial_reader.sim_arrhythmia_until = time.time() + 5.0

    def _on_waveform_type_change(self, val: str):
        """Actualiza el tipo de forma de onda en el simulador."""
        up = (val or "").upper()
        if ("BRADY" in up) or ("BRADI" in up):
            self.serial_reader.sim_waveform_type = "BRADYCARDIA"
        elif ("TACHY" in up) or ("TAQUI" in up):
            self.serial_reader.sim_waveform_type = "TACHYCARDIA"
        else:
            self.serial_reader.sim_waveform_type = "NORMAL"

    # ==============================================================
    # ── BUCLE PRINCIPAL DE ACTUALIZACION ─────────────────────────
    # ==============================================================

    def update_gui(self):
        """
        Bucle principal de refresco de la GUI.

        Dos capas de actualizacion:
          - RAPIDA (cada frame, ~80ms): grafico matplotlib, deteccion de picos.
          - LENTA  (cada 5 frames, ~400ms): badges del sidebar, status bar.
        """
        if not self.is_running:
            return

        try:
            self._update_gui_impl()
        except Exception:
            # No cortar el loop: si no se reprograma el `after`, Tk queda congelado.
            try:
                import traceback
                traceback.print_exc()
            except Exception:
                pass
        finally:
            refresh = max(40, self._safe_int(self.refresh_interval_var, 80))
            self.after(refresh, self.update_gui)

    def _update_gui_impl(self):
        """Implementación del refresco GUI (se llama desde `update_gui`)."""

        now     = time.time()

        # Separar actualizaciones rapidas (plot) de lentas (widgets sidebar)
        self._frame_count += 1
        do_slow = (self._frame_count % 5 == 0)

        win     = max(100, self._safe_int(self.app_state.window_size, 2000))
        gain    = max(0.1, self._safe_float(self.app_state.ecg_gain, 1.0))
        y_max_v = max(0.3, self._safe_float(self.app_state.y_max, 2.0))
        in_blank = now < getattr(self.app_state, "blank_until", 0.0)

        # ── Snapshot thread-safe del buffer ──────────────────────
        with self.app_state.data_lock:
            x_buf = list(self.app_state.time_buffer)
            y_buf = list(self.app_state.voltage_buffer)
            sc    = int(self.app_state.sample_count)

        # ── Ventana de visualizacion ──────────────────────────────
        if len(x_buf) > 1:
            xw_raw = x_buf[-win:]
            yw     = np.array(y_buf[-win:], dtype=float)
        else:
            xw_raw = list(range(max(2, win)))
            yw     = np.zeros(max(2, win), dtype=float)

        sample_rate = float(getattr(config, "SAMPLE_RATE", 1000))
        xw_sec      = np.asarray(xw_raw, dtype=float) / sample_rate

        # ── Deteccion de señal activa ─────────────────────────────
        signal_ok = (not in_blank) and self._signal_present(yw)
        self._update_no_signal_state(signal_ok)
        no_sig = getattr(self.app_state, "no_signal", False)

        # ── Blanking o sin señal: solo actualizar plot ────────────
        if in_blank or no_sig:
            self.ecg_line.set_data(xw_sec, np.zeros_like(yw))
            self.peaks_line.set_data([], [])
            self.qrs_line.set_data([], [])
            self._pace_line_pos.set_visible(False)
            self._pace_line_neg.set_visible(False)
            if len(xw_sec) > 1:
                self.ax.set_xlim(xw_sec[0], xw_sec[-1])
            self.ax.set_ylim(-y_max_v, y_max_v)
            self.mpl_canvas.draw_idle()

            if do_slow:
                rhythm_d = "ASISTOLIA" if no_sig else "---"
                self._update_vital_signs(0, rhythm_d, self.app_state.qrs_detected_count, False)
                self._update_pacemaker_panel()
                self._update_connection_panel()
                self.sb_samples_lbl.config(text=f"{sc:,}")
                self.sb_bpm_lbl.config(text="---", fg=self.T["muted"])
                self.sb_rhythm_lbl.config(text=rhythm_d)

            return

        # ── Centrado DC y ganancia ────────────────────────────────
        n_dc       = min(len(yw), 500)
        dc_offset  = float(np.median(yw[-n_dc:]))
        y_centered = (yw - dc_offset) * gain

        # ── Recibir último resultado del hilo de análisis ─────────
        try:
            while True:
                peaks, qrs, bpm, rhythm = self._analysis_out_q.get_nowait()
                self._analysis_peaks = peaks or []
                self._analysis_qrs = qrs or []
                self._analysis_bpm = float(bpm or 0.0)
                self._analysis_rhythm = rhythm or "---"
        except Exception:
            pass

        # ── Spike manual de marcapasos ────────────────────────────
        spike_visible = now < getattr(self.app_state, "pace_alert_until", 0.0)

        if getattr(self.app_state, "pace_pulse_pending", False):
            dur_ms     = max(1.0, self._safe_float(self.pace_duration_ms_var, 4.0))
            amp        = max(0.1, self._safe_float(self.app_state.pace_amplitude_var, 1.0))
            y_centered, s_idx, e_idx = self._inject_biphasic_spike(y_centered, dur_ms, amp)
            self.app_state.pace_pulse_pending = False
            spike_visible = True
            if len(xw_sec) > e_idx > 0:
                self._spike_x_sec  = float(xw_sec[s_idx])
                self._spike_x2_sec = float(xw_sec[max(s_idx, e_idx - 1)])
            else:
                self._spike_x_sec  = xw_sec[-1] - dur_ms / 1000.0
                self._spike_x2_sec = xw_sec[-1]

        # ── Deteccion de spike automatico (derivada) ──────────────
        if not spike_visible:
            dy = np.diff(y_centered)
            if len(dy) > 0:
                idxs = np.where(np.abs(dy) > float(getattr(config, "PACE_DERIV_THRESHOLD", 0.6)))[0]
                if len(idxs) > 0:
                    hold = max(0.5, self._safe_float(self.pace_alert_hold_var, 1.5))
                    self.app_state.pace_alert_until = now + hold
                    spike_visible = True
                    si = int(idxs[0])
                    if si < len(xw_sec):
                        self._spike_x_sec  = float(xw_sec[si])
                        half_i = min(si + max(1, len(xw_sec) // 60), len(xw_sec) - 1)
                        self._spike_x2_sec = float(xw_sec[half_i])

        # ── Enviar trabajo al hilo de análisis (no bloqueante) ─────
        r_thr  = max(0.01, self._safe_float(self.app_state.r_threshold, 0.3))
        r_dist = max(10,   self._safe_int(self.app_state.r_distance, 200))
        if self._analysis_running and self._analysis_in_q.empty():
            try:
                self._analysis_in_q.put_nowait((y_centered.copy(), sample_rate, r_thr, r_dist))
            except Exception:
                pass

        peaks  = list(self._analysis_peaks or [])
        bpm    = float(self._analysis_bpm or 0.0)
        rhythm = str(self._analysis_rhythm or "---")

        if bpm > 0:
            self.app_state.last_bpm = bpm

        # Contar nuevos QRS sin doble conteo en ventana deslizante
        for pi in peaks:
            if pi < len(xw_raw):
                abs_idx = int(xw_raw[pi])
                if abs_idx > self._last_qrs_abs_idx:
                    self.app_state.qrs_detected_count += 1
                    self._last_qrs_abs_idx = abs_idx
        self._last_qrs_complexes = list(self._analysis_qrs or [])

        # ── Actualizar artistas matplotlib ────────────────────────
        self.ecg_line.set_data(xw_sec, y_centered)
        self.ax.set_xlim(xw_sec[0], xw_sec[-1])
        self.ax.set_ylim(-y_max_v, y_max_v)

        if peaks:
            self.peaks_line.set_data(
                [xw_sec[i] for i in peaks if i < len(xw_sec)],
                [y_centered[i] for i in peaks if i < len(y_centered)],
            )
        else:
            self.peaks_line.set_data([], [])

        # Resaltado QRS: segmentos de la linea gruesa
        y_qrs = np.full(len(y_centered), np.nan)
        for qrs in self._last_qrs_complexes:
            o, f = qrs["onset"], qrs["offset"]
            y_qrs[o:f + 1] = y_centered[o:f + 1]
        self.qrs_line.set_data(xw_sec, y_qrs)

        # Marcadores de spike: dos axvline simples (sin manipulacion de poligonos)
        if spike_visible and self._spike_x_sec is not None:
            x1   = self._spike_x_sec
            xmid = self._spike_x2_sec if self._spike_x2_sec else x1 + 0.010
            self._pace_line_pos.set_xdata([x1,   x1])
            self._pace_line_neg.set_xdata([xmid, xmid])
            self._pace_line_pos.set_visible(True)
            self._pace_line_neg.set_visible(True)
        else:
            self._pace_line_pos.set_visible(False)
            self._pace_line_neg.set_visible(False)
            self._spike_x_sec  = None
            self._spike_x2_sec = None

        self.mpl_canvas.draw_idle()

        # ── Actualizaciones lentas del sidebar (~400ms) ───────────
        if do_slow:
            self._update_vital_signs(bpm, rhythm,
                                     self.app_state.qrs_detected_count, signal_ok)
            self._update_pacemaker_panel()
            self._update_connection_panel()
            self._update_simulation_panel()
            self.sb_samples_lbl.config(text=f"{sc:,}")
            bpm_color = (self.T["success"] if 60 <= bpm <= 100
                         else self.T["warning"] if bpm > 0 else self.T["muted"])
            self.sb_bpm_lbl.config(
                text=f"{bpm:.0f}" if bpm > 0 else "---", fg=bpm_color
            )
            self.sb_rhythm_lbl.config(text=rhythm)

        return

    # ==============================================================
    # ── CONTROL DE MODO AUTO ──────────────────────────────────────
    # ==============================================================

    def check_auto_mode(self):
        """Gestiona el cambio automatico de derivadas en modo AUTO."""
        if not self.is_running:
            return

        now = time.time()

        if not self.auto_scan_active:
            # Verificar si el timeout de inactividad activa el AUTO
            if self.app_state.operation_mode.get() == config.MODE_MANUAL:
                idle = now - self.app_state.last_manual_action_time
                if idle >= config.AUTO_TIMEOUT:
                    self.app_state.operation_mode.set(config.MODE_AUTO)
                    self.auto_scan_active = True
                    self.auto_scan_btn.config(text="AUTO SCAN  ON ",
                                              bg=self.T["accent"])
                    self.last_auto_change_time = now
        else:
            interval = max(1.0, self._safe_float(self.auto_switch_interval_var, 8.0))
            if (now - self.last_auto_change_time) >= interval:
                prev = self.app_state.current_mux_state
                self.app_state.next_derivation()
                self._send_mux_if_changed(prev)
                self._update_lead_buttons()
                self.last_auto_change_time = now

        self.after(300, self.check_auto_mode)

    def _send_mux_if_changed(self, previous_state: int):
        """Envia comando MUX al ESP32 si la derivada cambio, con blanking."""
        current = self.app_state.current_mux_state
        if current == previous_state:
            return

        self.serial_reader.send_mux_command(current)

        blank_sec = float(getattr(config, "DERIVATION_SWITCH_BLANK_SEC", 2.5))
        self.app_state.blank_until = time.time() + blank_sec

        with self.app_state.data_lock:
            self.app_state.voltage_buffer.clear()
            self.app_state.time_buffer.clear()

        self.app_state.no_signal_since = None
        self.app_state.no_signal       = False

    # ==============================================================
    # ── CIERRE DE LA APLICACION ───────────────────────────────────
    # ==============================================================

    def on_closing(self):
        """Cierra la aplicacion de forma limpia."""
        self.is_running = False
        self._analysis_running = False

        try:
            self.serial_reader.stop()
        except Exception:
            pass

        try:
            plt.close(self.fig)
        except Exception:
            pass

        self.destroy()
