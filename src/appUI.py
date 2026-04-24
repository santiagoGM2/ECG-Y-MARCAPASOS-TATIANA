"""
appUI.py  [REDISEÑADO — Clinical Light Theme]

CAMBIOS VISUALES (logica de señal, deteccion y serial INTACTAS):
  - Tema: ICU Dark  →  Clinical Light (blanco, navy, azul, ambar)
  - Layout: Header+Sidebar-derecho  →  Topbar+ECG-fullwidth+Tira-metricas+Panel-pestanas+Status
  - Derivaciones: panel lateral scrollable  →  botones en topbar horizontal
  - Controles: 6 paneles verticales  →  5 pestanas horizontales en la parte inferior
  - Colores ECG: fondo oscuro+cyan  →  fondo blanco+azul fuerte
  - Tipografia: misma fuente Segoe UI, pesos y tamanos reorganizados

Tema visual: Clinical Light — fondo blanco/gris claro, waveform azul, topbar azul marino.
Layout    : Topbar | ECG (full width, expansible) | Metricas | Pestanas | Status bar
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
# ----------- TEMA CLINICAL LIGHT (reemplaza ICU Dark) ----
# =========================================================

LIGHT_CLINICAL_THEME = {
    "name":           "Clinical Light",
    # Cromo general
    "bg":             "#EEF2F7",
    "panel":          "#FFFFFF",
    "border":         "#CBD5E6",
    "text":           "#1E293B",
    "muted":          "#64748B",
    "title":          "#0F172A",
    # Acciones
    "primary":        "#2563EB",
    "primary_active": "#1D4ED8",
    "accent":         "#0891B2",
    "accent_active":  "#0E7490",
    "success":        "#059669",
    "warning":        "#D97706",
    "danger":         "#DC2626",
    "danger_active":  "#B91C1C",
    # Fondos de badge
    "neutral_bg":     "#F1F5F9",
    "neutral_fg":     "#475569",
    "info_bg":        "#EFF6FF",
    "info_fg":        "#1D4ED8",
    "accent_bg":      "#ECFEFF",
    "accent_fg":      "#0E7490",
    "success_bg":     "#ECFDF5",
    "success_fg":     "#065F46",
    "warning_bg":     "#FFFBEB",
    "warning_fg":     "#92400E",
    "danger_bg":      "#FEF2F2",
    "danger_fg":      "#991B1B",
    "button_text":    "#FFFFFF",
    # Grafico ECG — colores completamente distintos al original
    "plot_bg":        "#FAFCFF",
    "plot_border":    "#C5D4EC",
    "plot_text":      "#334155",
    "grid":           "#DCE8F5",
    "ecg_line":       "#1A56DB",       # azul fuerte (original: cyan #22D3EE)
    "qrs_highlight":  "#3B82F6",       # azul medio para QRS
    "pace_spike":     "#D97706",       # ambar positivo (original: naranja #F59E0B)
    "peak":           "#DC2626",       # rojo marcadores R
    "baseline":       "#94A3B8",
    # Topbar
    "topbar_bg":      "#1E3A5F",
    "topbar_text":    "#F0F6FF",
    "topbar_muted":   "#93C5FD",
    "topbar_sep":     "#2D5A8E",
    "topbar_btn":     "#2D5A8E",
    # Pestanas
    "tab_active_bg":  "#1E3A5F",
    "tab_active_fg":  "#FFFFFF",
    "tab_idle_bg":    "#DDE7F5",
    "tab_idle_fg":    "#1E3A5F",
    "tab_bar_bg":     "#C7D8EF",
}

_LEAD_NAMES = ["I", "II", "III", "aVR", "aVL", "aVF"]


class ECGApp(tk.Tk):
    """
    Aplicacion principal de monitoreo ECG + Marcapasos.
    Gestiona el estado compartido (AppState), el lector serial/simulacion
    (SerialReader) y toda la interfaz grafica.
    """

    def __init__(self):
        super().__init__()

        self.T = LIGHT_CLINICAL_THEME  # alias corto para el tema

        self.title("ECG Monitor — Análisis Cardiaco en Tiempo Real")
        self.geometry("1400x860")
        self.minsize(1100, 760)
        self.configure(bg=self.T["bg"])

        self.is_running = True

        # ── Estado compartido y lector serial ─────────────────────
        self.app_state = AppState(master=self)

        # Iniciar siempre en simulacion al abrir la app.
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
        ports        = list_available_ports()
        default_port = config.SERIAL_PORT if not ports else (
            config.SERIAL_PORT if config.SERIAL_PORT in ports else ports[0]
        )
        self.port_var = tk.StringVar(value=default_port)
        self.baud_var = tk.StringVar(value=str(config.BAUDRATE))

        # ── Variables de simulacion ───────────────────────────────
        self.sim_hr_var    = tk.DoubleVar(value=float(getattr(config, "SIMULATION_HEART_RATE", 72)))
        self.sim_amp_var   = tk.DoubleVar(value=1.0)
        self.sim_noise_var = tk.DoubleVar(value=float(getattr(config, "SIMULATION_NOISE", 0.02)))
        self.sim_wf_var    = tk.StringVar(value="ECG NORMAL")

        # ── Estado interno del marcapasos visual ──────────────────
        self._spike_x_sec  = None
        self._spike_x2_sec = None

        self._frame_count        = 0
        self._last_qrs_abs_idx   = 0
        self._last_qrs_complexes = []
        self._vital_cache        = None
        self._last_rr_ms         = 0.0

        # ── Variables de visualizacion configurables ──────────────
        self.show_grid_var     = tk.BooleanVar(value=True)
        self.show_peaks_var    = tk.BooleanVar(value=True)
        self.show_qrs_var      = tk.BooleanVar(value=True)
        self.show_baseline_var = tk.BooleanVar(value=True)
        self.autoscale_y_var   = tk.BooleanVar(value=False)
        self.session_label_var = tk.StringVar(value="DEMO BIOMÉDICA")
        self._pacing_mode_var  = tk.StringVar(value="VOO")
        self._ecg_color_idx    = 0
        self._ecg_color_presets = ["#1A56DB", "#059669", "#DC2626", "#7C3AED", "#0891B2"]

        # ── Refs a widgets dinamicos ──────────────────────────────
        self._lead_buttons = {}
        self._qrs_spans    = []

        # ── Construccion de la interfaz ───────────────────────────
        self._create_widgets()

        # ── Actualizaciones iniciales ─────────────────────────────
        self._update_lead_buttons()
        self._update_connection_panel()
        self._update_pacemaker_panel()
        self._update_clock()
        self.after(300, self._draw_biphasic_preview)

        # ── Traces para sincronizar parametros de simulacion ──────
        for _v in (self.sim_hr_var, self.sim_amp_var, self.sim_noise_var,
                   self.app_state.pace_amplitude_var, self.app_state.pace_bpm_var):
            _v.trace_add("write", self._sync_sim_params)

        # ── Arrancar hilo serial / simulacion ────────────────────
        self.serial_reader.start()

        # ── Programar bucles de actualizacion ────────────────────
        self.after(100, self.update_gui)
        self.after(300, self.check_auto_mode)

        self.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.report_callback_exception = self._on_tk_exception

        # ── Hilo de analisis (picos R / QRS / BPM) ───────────────
        self._analysis_in_q    = queue.Queue(maxsize=1)
        self._analysis_out_q   = queue.Queue(maxsize=1)
        self._analysis_running = True
        self._analysis_thread  = threading.Thread(
            target=self._analysis_loop, name="ECGAnalysisWorker", daemon=True
        )
        self._analysis_thread.start()

        self._analysis_peaks  = []
        self._analysis_qrs    = []
        self._analysis_bpm    = 0.0
        self._analysis_rhythm = "---"

    # ==============================================================
    # ── HELPERS DE EXCEPCION Y ANALISIS (sin cambios) ─────────────
    # ==============================================================

    def _on_tk_exception(self, exc, val, tb):
        """Evita que una excepcion rompa el loop after y congele la GUI."""
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
        """Hilo de analisis para evitar bloqueos del hilo de GUI."""
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
                peaks  = detect_r_peaks(y_centered, r_thr, r_dist)
                bpm    = calculate_bpm(peaks, sample_rate)
                rhythm = classify_rhythm(bpm)
                qrs    = detect_qrs_complex(y_centered, peaks, sample_rate)
                result = (peaks, qrs, bpm, rhythm)
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
                continue

    # ==============================================================
    # ── HELPERS SEGUROS (sin cambios) ─────────────────────────────
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
        """Sincroniza parametros de simulacion con SerialReader (via trace, no en cada frame)."""
        try:
            self.serial_reader.sim_heart_rate  = self._safe_float(self.sim_hr_var,    72.0)
            self.serial_reader.sim_amplitude   = self._safe_float(self.sim_amp_var,    1.0)
            self.serial_reader.sim_noise_level = self._safe_float(self.sim_noise_var,  0.02)
            self.serial_reader.pace_amplitude  = self._safe_float(self.app_state.pace_amplitude_var, 1.0)
            self.serial_reader.pace_bpm        = self._safe_float(self.app_state.pace_bpm_var,       60.0)
        except Exception:
            pass

    def _topbar_toggle_hw(self):
        """Alterna entre modo hardware y simulación desde el topbar."""
        if self.app_state.esp32_connected:
            self._restart_reader(port="NONE_DISCONNECT")
            self.after(500, self._update_topbar_hw_state)
        else:
            port = self.port_var.get().strip()
            if not port or port == "NONE_SIM":
                return
            try:
                baud = int(self.baud_var.get())
            except Exception:
                baud = config.BAUDRATE
            config.SERIAL_PORT = port
            config.BAUDRATE    = baud
            self.topbar_hw_btn.config(text="Conectando…", bg=self.T["muted"])
            self._restart_reader(port=port)
            self.after(1200, self._update_topbar_hw_state)

    def _topbar_refresh_ports(self):
        """Actualiza la lista de puertos en el topbar."""
        ports = list_available_ports()
        if not ports:
            return
        menu = self.topbar_port_menu["menu"]
        menu.delete(0, "end")
        for p in ports:
            menu.add_command(label=p, command=lambda v=p: self.port_var.set(v))
        if self.port_var.get() not in ports:
            self.port_var.set(ports[0])
        # Sincronizar también el menú de la pestaña CONEXIÓN
        try:
            self.on_refresh_ports()
        except Exception:
            pass

    def _update_topbar_hw_state(self):
        """Actualiza el botón del topbar según el estado de conexión."""
        if self.app_state.esp32_connected:
            self.topbar_hw_btn.config(
                text="✖ DESCONECTAR",
                bg=self.T["danger"],
                activebackground=self.T["danger_active"],
            )
            self._set_badge(self.mode_badge_hdr, "HARDWARE", "success")
        else:
            self.topbar_hw_btn.config(
                text="⚡ CONECTAR",
                bg="#D97706",
                activebackground="#B45309",
            )
            self._set_badge(self.mode_badge_hdr, "SIMULACIÓN", "warning")

    def _set_ecg_color(self, color: str):
        """Cambia el color de la línea ECG en tiempo real."""
        try:
            self.ecg_line.set_color(color)
            self.T["ecg_line"] = color
            self.mpl_canvas.draw_idle()
        except Exception:
            pass

    def _on_pacing_mode_change(self, *_):
        """Actualiza la descripcion del modo de estimulacion seleccionado."""
        descriptions = {
            "VOO": "Asincrónico ventricular\n(sin sensing, pace fijo)",
            "VVI": "Inhibido ventricular\n(inhibe si hay R propio)",
            "AOO": "Asincrónico auricular\n(estimula aurícula fija)",
            "AAI": "Inhibido auricular\n(inhibe si hay P propio)",
        }
        mode = self._pacing_mode_var.get()
        if hasattr(self, "pace_mode_desc"):
            self.pace_mode_desc.config(text=descriptions.get(mode, ""))

    # ==============================================================
    # ── LAYOUT PRINCIPAL (rediseñado) ─────────────────────────────
    # ==============================================================

    def _create_widgets(self):
        """
        Nueva estructura de layout:
          Topbar (fijo, navy) → ECG plot (full width, expansible) →
          Tira de metricas (fija, 86px) → Panel de pestanas (fijo, 220px) →
          Status bar (fijo, 26px, navy)
        """
        root = tk.Frame(self, bg=self.T["bg"])
        root.pack(fill=tk.BOTH, expand=True)

        # 1. Topbar: titulo + botones de derivacion + badges + reloj
        self._create_topbar(root)

        # 2. Grafico ECG — altura fija (~25 % menor que a pantalla completa)
        ecg_outer = tk.Frame(
            root, bg=self.T["panel"],
            highlightthickness=1, highlightbackground=self.T["border"], bd=0,
            height=300,
        )
        ecg_outer.pack(fill=tk.X, padx=8, pady=(4, 0))
        ecg_outer.pack_propagate(False)
        self._create_ecg_plot(ecg_outer)

        # 3. Tira de metricas (5 tarjetas)
        metrics_frame = tk.Frame(root, bg=self.T["bg"], height=66)
        metrics_frame.pack(fill=tk.X, padx=8, pady=(4, 0))
        metrics_frame.pack_propagate(False)
        self._create_metrics_strip(metrics_frame)

        # 4. Panel de pestanas inferior — más alto gracias al ECG reducido
        tabs_frame = tk.Frame(root, bg=self.T["bg"], height=272)
        tabs_frame.pack(fill=tk.BOTH, expand=True, padx=8, pady=(4, 0))
        tabs_frame.pack_propagate(False)
        self._create_tab_panel(tabs_frame)

        # 5. Status bar (delgada, navy, misma estetica que topbar)
        self._create_status_bar(root)

    # ----------------------------------------------------------
    def _create_topbar(self, parent):
        """
        Barra superior navy: [Logo | Derivaciones | AUTO] [Badges + Reloj]
        Los 6 botones de derivacion viven aqui (vs panel lateral en el original).
        """
        bar = tk.Frame(parent, bg=self.T["topbar_bg"])
        bar.pack(fill=tk.X)

        # — Seccion izquierda: logo / titulo —
        left = tk.Frame(bar, bg=self.T["topbar_bg"])
        left.pack(side=tk.LEFT, padx=(14, 0), pady=10)

        tk.Label(
            left, text="ECG MONITOR",
            bg=self.T["topbar_bg"], fg=self.T["topbar_text"],
            font=("Segoe UI", 13, "bold"),
        ).pack(anchor="w")
        tk.Label(
            left, text="Análisis cardiaco en tiempo real",
            bg=self.T["topbar_bg"], fg=self.T["topbar_muted"],
            font=("Segoe UI", 7),
        ).pack(anchor="w")

        tk.Frame(bar, bg=self.T["topbar_sep"], width=1).pack(
            side=tk.LEFT, fill=tk.Y, padx=14, pady=8
        )

        # — Seccion central: botones de derivacion —
        center = tk.Frame(bar, bg=self.T["topbar_bg"])
        center.pack(side=tk.LEFT, pady=10)

        tk.Label(
            center, text="DERIVACIÓN",
            bg=self.T["topbar_bg"], fg=self.T["topbar_muted"],
            font=("Segoe UI", 7, "bold"),
        ).pack(anchor="w")

        btn_row = tk.Frame(center, bg=self.T["topbar_bg"])
        btn_row.pack(anchor="w")

        for name, state in zip(_LEAD_NAMES, range(6)):
            btn = tk.Button(
                btn_row, text=name,
                command=lambda s=state: self.on_lead_select(s),
                bg=self.T["topbar_btn"], fg=self.T["topbar_text"],
                activebackground=self.T["primary"], activeforeground="#FFFFFF",
                relief="flat", bd=0, cursor="hand2",
                font=("Segoe UI", 9, "bold"),
                padx=11, pady=4, highlightthickness=0,
            )
            btn.pack(side=tk.LEFT, padx=(0, 3))
            self._lead_buttons[state] = btn

        # — Seccion derecha: control hardware + reloj —
        right = tk.Frame(bar, bg=self.T["topbar_bg"])
        right.pack(side=tk.RIGHT, padx=(0, 14), pady=8)

        # Reloj
        self.clock_label = tk.Label(
            right, text="--:--:--",
            bg=self.T["topbar_bg"], fg=self.T["topbar_text"],
            font=("Segoe UI", 13, "bold"),
        )
        self.clock_label.pack(side=tk.RIGHT, padx=(10, 0))

        tk.Frame(bar, bg=self.T["topbar_sep"], width=1).pack(
            side=tk.RIGHT, fill=tk.Y, padx=8, pady=6
        )

        # Badge de modo (unico, sin duplicado)
        self.mode_badge_hdr = tk.Label(
            right, text="SIMULACIÓN",
            font=("Segoe UI", 8, "bold"), padx=9, pady=4, bd=0,
        )
        self.mode_badge_hdr.pack(side=tk.RIGHT, padx=(0, 6))
        self._set_badge(self.mode_badge_hdr, "SIMULACIÓN", "warning")

        tk.Frame(bar, bg=self.T["topbar_sep"], width=1).pack(
            side=tk.RIGHT, fill=tk.Y, padx=8, pady=6
        )

        # ── Zona de conexion hardware (prominente en topbar) ──────
        hw_zone = tk.Frame(bar, bg=self.T["topbar_bg"])
        hw_zone.pack(side=tk.RIGHT, padx=(0, 4), pady=7)

        tk.Label(
            hw_zone, text="HARDWARE:",
            bg=self.T["topbar_bg"], fg=self.T["topbar_muted"],
            font=("Segoe UI", 7, "bold"),
        ).pack(side=tk.LEFT, padx=(0, 4))

        # Dropdown de puerto COM (reutiliza self.port_var del __init__)
        _ports = list_available_ports() or [config.SERIAL_PORT]
        if _ports and self.port_var.get() not in _ports:
            self.port_var.set(_ports[0])
        self.topbar_port_menu = tk.OptionMenu(hw_zone, self.port_var, *_ports)
        self.topbar_port_menu.configure(
            bg=self.T["topbar_btn"], fg=self.T["topbar_text"],
            activebackground=self.T["primary"], activeforeground="#FFFFFF",
            relief="flat", bd=0, font=("Segoe UI", 8), width=7,
            highlightthickness=0,
        )
        self.topbar_port_menu["menu"].configure(
            bg=self.T["neutral_bg"], fg=self.T["text"],
            activebackground=self.T["primary"], activeforeground="#FFFFFF",
        )
        self.topbar_port_menu.pack(side=tk.LEFT, padx=(0, 3))

        # Botón refrescar puertos
        tk.Button(
            hw_zone, text="↻",
            command=self._topbar_refresh_ports,
            bg=self.T["topbar_btn"], fg=self.T["topbar_text"],
            activebackground=self.T["primary"], activeforeground="#FFFFFF",
            relief="flat", bd=0, cursor="hand2",
            font=("Segoe UI", 9), padx=5, pady=3, highlightthickness=0,
        ).pack(side=tk.LEFT, padx=(0, 4))

        # Botón principal: CONECTAR / DESCONECTAR
        self.topbar_hw_btn = tk.Button(
            hw_zone, text="⚡ CONECTAR",
            command=self._topbar_toggle_hw,
            bg="#D97706", fg="#FFFFFF",
            activebackground="#B45309", activeforeground="#FFFFFF",
            relief="flat", bd=0, cursor="hand2",
            font=("Segoe UI", 8, "bold"), padx=10, pady=5, highlightthickness=0,
        )
        self.topbar_hw_btn.pack(side=tk.LEFT)

        # Referencia de compatibilidad (usada por _update_connection_panel)
        self.conn_badge_hdr = self.mode_badge_hdr

    # ----------------------------------------------------------
    def _create_ecg_plot(self, parent):
        """
        Grafico matplotlib con tema claro.
        Fondo blanco, waveform azul, grid azul claro, picos rojo, pace ambar+violeta.
        """
        # Barra compacta: derivación activa + leyenda de colores
        title_bar = tk.Frame(parent, bg=self.T["panel"])
        title_bar.pack(fill=tk.X, padx=10, pady=(4, 2))

        tk.Label(
            title_bar, text="ECG —",
            bg=self.T["panel"], fg=self.T["muted"],
            font=("Segoe UI", 8, "bold"),
        ).pack(side=tk.LEFT)

        self.lead_title_label = tk.Label(
            title_bar, text="Derivación II",
            bg=self.T["panel"], fg=self.T["primary"],
            font=("Segoe UI", 9, "bold"),
        )
        self.lead_title_label.pack(side=tk.LEFT, padx=(4, 0))

        # Leyenda de colores (compacta)
        legend_frame = tk.Frame(title_bar, bg=self.T["panel"])
        legend_frame.pack(side=tk.RIGHT)
        for col, lbl in [
            (self.T["ecg_line"],      "ECG"),
            (self.T["peak"],          "Pico R"),
            (self.T["qrs_highlight"], "QRS"),
            (self.T["pace_spike"],    "Pace +"),
            ("#7C3AED",               "Pace −"),
        ]:
            tk.Label(legend_frame, text="●", bg=self.T["panel"], fg=col,
                     font=("Segoe UI", 9)).pack(side=tk.LEFT, padx=(5, 1))
            tk.Label(legend_frame, text=lbl, bg=self.T["panel"], fg=self.T["muted"],
                     font=("Segoe UI", 7)).pack(side=tk.LEFT, padx=(0, 3))

        # ── Figura matplotlib (colores claros) ────────────────────
        self.fig, self.ax = plt.subplots(figsize=(10, 2.8), dpi=96)
        self.fig.patch.set_facecolor(self.T["panel"])
        self.ax.set_facecolor(self.T["plot_bg"])

        for sp in self.ax.spines.values():
            sp.set_color(self.T["plot_border"])

        self.ax.tick_params(colors=self.T["plot_text"], labelsize=8)
        self.ax.set_xlabel("Tiempo (s)", color=self.T["plot_text"], fontsize=8)
        self.ax.set_ylabel("mV", color=self.T["plot_text"], fontsize=8)
        self.ax.grid(True, color=self.T["grid"], alpha=0.9, linewidth=0.6,
                     linestyle="-", which="both")
        self.ax.margins(x=0)

        # Baseline
        self.baseline_line = self.ax.axhline(
            0, color=self.T["baseline"], linewidth=0.8, alpha=0.6, zorder=1
        )

        # Marcadores de spike (dos axvline: fase+ ambar, fase- violeta)
        self._pace_line_pos = self.ax.axvline(
            x=0, color=self.T["pace_spike"], linewidth=2.5,
            linestyle="--", visible=False, zorder=6, alpha=0.9, label="Pace +"
        )
        self._pace_line_neg = self.ax.axvline(
            x=0, color="#7C3AED", linewidth=2.5,
            linestyle="--", visible=False, zorder=6, alpha=0.9, label="Pace -"
        )

        # Waveform ECG (azul fuerte, vs cyan original)
        self.ecg_line, = self.ax.plot(
            [], [], linewidth=1.8, color=self.T["ecg_line"],
            solid_capstyle="round", zorder=4
        )

        # Resaltado QRS
        self.qrs_line, = self.ax.plot(
            [], [], linewidth=6, color=self.T["qrs_highlight"],
            alpha=0.25, zorder=3, solid_capstyle="round"
        )

        # Picos R (triangulos rojos hacia abajo, vs circulos del original)
        self.peaks_line, = self.ax.plot(
            [], [], linestyle="", marker="v", markersize=7,
            color=self.T["peak"], zorder=7, markeredgewidth=0
        )

        self.fig.subplots_adjust(left=0.06, right=0.99, top=0.90, bottom=0.16)

        self.mpl_canvas = FigureCanvasTkAgg(self.fig, master=parent)
        mpl_widget = self.mpl_canvas.get_tk_widget()
        mpl_widget.pack(fill=tk.BOTH, expand=True, padx=4, pady=(2, 4))
        mpl_widget.configure(bg=self.T["panel"], highlightthickness=0)

    # ----------------------------------------------------------
    def _create_metrics_strip(self, parent):
        """
        Tira de 4 tarjetas horizontales: BPM | Ritmo | QRS | Señal.
        Reemplaza el panel 'Signos Vitales' del sidebar original.
        """
        def _card(bg_accent):
            card = tk.Frame(
                parent, bg=self.T["panel"],
                highlightthickness=2, highlightbackground=bg_accent, bd=0,
            )
            card.pack(side=tk.LEFT, fill=tk.BOTH, expand=True,
                      padx=(0, 6), pady=2)
            return card

        # — Tarjeta BPM —
        c1 = _card(self.T["primary"])
        tk.Label(c1, text="FREC. CARDIACA",
                 bg=self.T["panel"], fg=self.T["muted"],
                 font=("Segoe UI", 7, "bold")).pack(anchor="center", pady=(4, 0))
        self.bpm_big_label = tk.Label(
            c1, text="---",
            bg=self.T["panel"], fg=self.T["success"],
            font=("Segoe UI", 22, "bold"),
        )
        self.bpm_big_label.pack(anchor="center")
        tk.Label(c1, text="lat/min",
                 bg=self.T["panel"], fg=self.T["muted"],
                 font=("Segoe UI", 7)).pack(anchor="center", pady=(0, 2))

        # — Tarjeta Ritmo —
        c2 = _card(self.T["success"])
        tk.Label(c2, text="RITMO",
                 bg=self.T["panel"], fg=self.T["muted"],
                 font=("Segoe UI", 7, "bold")).pack(anchor="center", pady=(6, 0))
        self.rhythm_badge = tk.Label(
            c2, text="ASISTOLIA", padx=12, pady=5, bd=0,
            font=("Segoe UI", 11, "bold"),
        )
        self.rhythm_badge.pack(fill=tk.X, padx=10, pady=4)
        self._set_badge(self.rhythm_badge, "ASISTOLIA", "neutral",
                        font=("Segoe UI", 11, "bold"))

        # — Tarjeta QRS —
        c3 = _card(self.T["accent"])
        tk.Label(c3, text="COMPLEJOS QRS",
                 bg=self.T["panel"], fg=self.T["muted"],
                 font=("Segoe UI", 7, "bold")).pack(anchor="center", pady=(6, 0))
        self.qrs_count_badge = tk.Label(
            c3, text="0", padx=12, pady=5, bd=0,
            font=("Segoe UI", 11, "bold"),
        )
        self.qrs_count_badge.pack(fill=tk.X, padx=10, pady=4)
        self._set_badge(self.qrs_count_badge, "0", "info")

        # — Tarjeta Señal —
        c4 = _card(self.T["warning"])
        tk.Label(c4, text="CALIDAD DE SEÑAL",
                 bg=self.T["panel"], fg=self.T["muted"],
                 font=("Segoe UI", 7, "bold")).pack(anchor="center", pady=(6, 0))
        self.sig_quality_badge = tk.Label(
            c4, text="---", padx=12, pady=5, bd=0,
            font=("Segoe UI", 11, "bold"),
        )
        self.sig_quality_badge.pack(fill=tk.X, padx=10, pady=4)
        self._set_badge(self.sig_quality_badge, "---", "neutral")

        # — Tarjeta Intervalo R-R —
        c5 = _card(self.T["accent"])
        tk.Label(c5, text="INTERVALO R-R",
                 bg=self.T["panel"], fg=self.T["muted"],
                 font=("Segoe UI", 7, "bold")).pack(anchor="center", pady=(6, 0))
        self.rr_interval_badge = tk.Label(
            c5, text="---", padx=10, pady=3, bd=0,
            font=("Segoe UI", 13, "bold"),
        )
        self.rr_interval_badge.pack(anchor="center")
        tk.Label(c5, text="ms",
                 bg=self.T["panel"], fg=self.T["muted"],
                 font=("Segoe UI", 7)).pack(anchor="center", pady=(0, 2))
        self._set_badge(self.rr_interval_badge, "---", "accent")

    # ----------------------------------------------------------
    def _create_tab_panel(self, parent):
        """
        Panel de pestanas horizontal en la parte inferior.
        5 pestanas: DERIVACIONES | MARCAPASOS | SEÑAL | CONEXIÓN | SIMULACIÓN
        Reemplaza el sidebar scrollable con 6 paneles del original.
        """
        # Barra de pestanas
        tab_bar = tk.Frame(parent, bg=self.T["tab_bar_bg"], height=30)
        tab_bar.pack(fill=tk.X)
        tab_bar.pack_propagate(False)

        # Area de contenido
        self._tab_content_area = tk.Frame(
            parent, bg=self.T["panel"],
            highlightthickness=1, highlightbackground=self.T["border"], bd=0,
        )
        self._tab_content_area.pack(fill=tk.BOTH, expand=True)

        self._tab_frames = {}
        self._tab_btns   = {}

        tab_defs = [
            ("DERIVACIONES", self._build_tab_derivaciones),
            ("MARCAPASOS",   self._build_tab_pacemaker),
            ("SEÑAL",        self._build_tab_signal),
            ("CONEXIÓN",     self._build_tab_connection),
            ("SIMULACIÓN",   self._build_tab_simulation),
            ("HARDWARE",     self._build_tab_hardware),
        ]

        for name, builder in tab_defs:
            # Frame de contenido (construido pero oculto hasta seleccion)
            frame = tk.Frame(self._tab_content_area, bg=self.T["panel"])
            self._tab_frames[name] = frame
            builder(frame)

            # Boton de pestana
            btn = tk.Label(
                tab_bar, text=f"  {name}  ",
                font=("Segoe UI", 8, "bold"),
                padx=4, pady=5, cursor="hand2",
            )
            btn.pack(side=tk.LEFT)
            btn.bind("<Button-1>", lambda *_, n=name: self._switch_tab(n))
            self._tab_btns[name] = btn

        # Mostrar primera pestana por defecto
        self._switch_tab("DERIVACIONES")

    def _switch_tab(self, name: str):
        """Muestra la pestana indicada y oculta el resto."""
        for n, frame in self._tab_frames.items():
            if n == name:
                frame.pack(fill=tk.BOTH, expand=True)
            else:
                frame.pack_forget()

        for n, btn in self._tab_btns.items():
            if n == name:
                btn.config(
                    bg=self.T["tab_active_bg"],
                    fg=self.T["tab_active_fg"],
                )
            else:
                btn.config(
                    bg=self.T["tab_idle_bg"],
                    fg=self.T["tab_idle_fg"],
                )

    # ==============================================================
    # ── CONSTRUCTORES DE PESTANAS ─────────────────────────────────
    # ==============================================================

    def _build_tab_derivaciones(self, parent):
        """Pestana DERIVACIONES: badge de derivada activa + auto scan."""
        # Col 1: derivada activa
        col1 = self._tab_col(parent, "DERIVADA ACTIVA")

        self.active_lead_badge = tk.Label(
            col1, text="DERIVACIÓN II", padx=12, pady=7, bd=0,
            font=("Segoe UI", 13, "bold"),
        )
        self.active_lead_badge.pack(fill=tk.X, pady=(0, 4))
        self._set_badge(self.active_lead_badge, "DERIVACIÓN II", "info",
                        font=("Segoe UI", 13, "bold"))

        # Col 2: escaneo automatico
        col2 = self._tab_col(parent, "ESCANEO AUTOMÁTICO")

        self.auto_scan_btn = self._btn(
            col2, "ESCANEO AUTO  APAGADO", self.on_auto_scan_toggle,
            kind="neutral", pady=8, font=("Segoe UI", 9, "bold"),
        )
        self.auto_scan_btn.pack(fill=tk.X, pady=(0, 6))

        r = self._row(col2, "Intervalo auto (s)", "Segundos entre cambios")
        self._spinbox(r, self.auto_switch_interval_var, 1.0, 30.0, 1.0, 7).pack()

        # Col 3: informacion
        col3 = self._tab_col(parent, "INFORMACIÓN", sep=False)
        tk.Label(
            col3,
            text="Selecciona la derivación\nusando los botones\ndel topbar superior.",
            bg=self.T["panel"], fg=self.T["muted"],
            font=("Segoe UI", 8), justify="left",
        ).pack(anchor="w")

    def _build_tab_pacemaker(self, parent):
        """Pestana MARCAPASOS: trigger, parametros, preview bifasico, auto-pacing."""
        # Col 1: estado y trigger
        col1 = self._tab_col(parent, "ESTADO / TRIGGER")

        self.pace_status_badge = tk.Label(
            col1, text="SIN ALERTA", padx=10, pady=6, bd=0,
            font=("Segoe UI", 10, "bold"),
        )
        self.pace_status_badge.pack(fill=tk.X, pady=(0, 6))
        self._set_badge(self.pace_status_badge, "SIN ALERTA", "success",
                        font=("Segoe UI", 10, "bold"))

        self._btn(
            col1, "DISPARAR PULSO", self.on_pace_trigger,
            kind="danger", pady=10, font=("Segoe UI", 11, "bold"),
        ).pack(fill=tk.X)

        # Col 2: parametros del pulso
        col2 = self._tab_col(parent, "PARÁMETROS DEL PULSO")

        r = self._row(col2, "Amplitud (V)")
        self._spinbox(r, self.app_state.pace_amplitude_var, 0.1, 3.0, 0.1, 7).pack()

        r = self._row(col2, "Frecuencia (BPM)")
        self._spinbox(r, self.app_state.pace_bpm_var, 30.0, 200.0, 1.0, 7).pack()

        r = self._row(col2, "Duración (ms)")
        self._spinbox(r, self.pace_duration_ms_var, 1.0, 30.0, 1.0, 7).pack()

        # Col 3: preview bifasico
        col3 = self._tab_col(parent, "FORMA DE ONDA BIFÁSICA")

        preview_frame = tk.Frame(
            col3, bg=self.T["plot_bg"],
            highlightthickness=1, highlightbackground=self.T["border"],
            height=62, width=210,
        )
        preview_frame.pack(fill=tk.X)
        preview_frame.pack_propagate(False)

        self.pace_canvas = tk.Canvas(
            preview_frame, bg=self.T["plot_bg"],
            highlightthickness=0, height=62,
        )
        self.pace_canvas.pack(fill=tk.BOTH, expand=True)
        self.pace_canvas.bind(
            "<Configure>", lambda _: self.after_idle(self._draw_biphasic_preview)
        )

        # Col 3b: modo de estimulacion
        col3b = self._tab_col(parent, "MODO MARCAPASOS")
        r_mode = self._row(col3b, "Modo")
        mode_options = ["VOO", "VVI", "AOO", "AAI"]
        mode_menu = tk.OptionMenu(r_mode, self._pacing_mode_var, *mode_options)
        mode_menu.configure(
            bg=self.T["neutral_bg"], fg=self.T["text"],
            activebackground=self.T["primary"], activeforeground="#FFFFFF",
            highlightthickness=1, highlightbackground=self.T["border"],
            relief="flat", font=("Segoe UI", 9), width=8,
        )
        mode_menu["menu"].configure(
            bg=self.T["neutral_bg"], fg=self.T["text"],
            activebackground=self.T["primary"], activeforeground="#FFFFFF",
        )
        mode_menu.pack()

        # Descripcion del modo
        self.pace_mode_desc = tk.Label(
            col3b, text="Asincrónico ventricular",
            bg=self.T["panel"], fg=self.T["muted"],
            font=("Segoe UI", 7), wraplength=130, justify="left",
        )
        self.pace_mode_desc.pack(anchor="w", pady=(4, 0))
        self._pacing_mode_var.trace_add("write", self._on_pacing_mode_change)

        self.pace_energy_badge = self._metric_row(col3b, "Energía pulso")
        self._set_badge(self.pace_energy_badge, "---", "warning")

        # Col 4: auto-pacing y badges de parametros
        col4 = self._tab_col(parent, "AUTO-ESTIMULACIÓN", sep=False)

        self.auto_pacing_var.trace_add("write", self._on_auto_pacing_change)
        chk = tk.Checkbutton(
            col4, text="Habilitar auto-estimulación",
            variable=self.auto_pacing_var,
            bg=self.T["panel"], fg=self.T["text"],
            selectcolor=self.T["neutral_bg"],
            activebackground=self.T["panel"],
            activeforeground=self.T["text"],
            font=("Segoe UI", 9),
        )
        chk.pack(anchor="w", pady=(0, 8))

        self.pace_amp_badge = self._metric_row(col4, "Amplitud activa")
        self.pace_dur_badge = self._metric_row(col4, "Duración activa")
        self.pace_bpm_badge = self._metric_row(col4, "Frecuencia activa")

    def _build_tab_signal(self, parent):
        """Pestana SEÑAL: umbrales de deteccion + ajustes de visualizacion (3 columnas)."""
        # Col 1: tasa de muestreo + deteccion R
        col1 = self._tab_col(parent, "DETECCIÓN DE PICOS R")

        self.acq_rate_badge = self._metric_row(col1, "Frec. de muestreo")
        self._set_badge(self.acq_rate_badge, f"{getattr(config, 'SAMPLE_RATE', 1000)} Hz", "info")

        r = self._row(col1, "Umbral R (V)", "Voltaje min. para detectar R")
        self._spinbox(r, self.app_state.r_threshold, 0.05, 3.0, 0.05, 7).pack()

        r = self._row(col1, "Distancia R (muestras)", "Min. muestras entre picos")
        self._spinbox(r, self.app_state.r_distance, 50, 600, 10, 7).pack()

        # Col 2: ganancia y ventana temporal
        col2 = self._tab_col(parent, "AMPLIFICACIÓN Y VENTANA")

        r = self._row(col2, "Ganancia ECG", "Amplificacion vertical de la señal")
        self._spinbox(r, self.app_state.ecg_gain, 0.1, 10.0, 0.1, 7).pack()

        r = self._row(col2, "Ventana (muestras)", "Muestras visibles en el grafico")
        self._spinbox(r, self.app_state.window_size, 200, 5000, 100, 7).pack()

        # Col 3: escala Y y refresco
        col3 = self._tab_col(parent, "ESCALA Y REFRESCO")

        r = self._row(col3, "Y máx (V)", "Limite del eje vertical")
        self._spinbox(r, self.app_state.y_max, 0.5, 10.0, 0.1, 7).pack()

        r = self._row(col3, "Refresco (ms)", "Intervalo de actualizacion de GUI")
        self._spinbox(r, self.refresh_interval_var, 20, 500, 10, 7).pack()

        self.acq_stats_badge = self._metric_row(col3, "Frec. muestreo")
        self._set_badge(self.acq_stats_badge, f"{getattr(config, 'SAMPLE_RATE', 1000)} Hz", "info")

        # Col 4: opciones de visualizacion y sesion
        col4 = self._tab_col(parent, "VISUALIZACIÓN", sep=False, min_width=190)

        _chk_opts = [
            ("Cuadrícula ECG",   self.show_grid_var),
            ("Picos R (▼)",      self.show_peaks_var),
            ("Resalte QRS",      self.show_qrs_var),
            ("Línea base",       self.show_baseline_var),
            ("Auto-escala Y",    self.autoscale_y_var),
        ]
        for txt, var in _chk_opts:
            tk.Checkbutton(
                col4, text=txt, variable=var,
                bg=self.T["panel"], fg=self.T["text"],
                selectcolor=self.T["neutral_bg"],
                activebackground=self.T["panel"],
                activeforeground=self.T["text"],
                font=("Segoe UI", 8),
            ).pack(anchor="w", pady=1)

        tk.Frame(col4, bg=self.T["border"], height=1).pack(fill=tk.X, pady=5)

        # Color de la línea ECG
        color_row = tk.Frame(col4, bg=self.T["panel"])
        color_row.pack(fill=tk.X, pady=(0, 4))
        tk.Label(color_row, text="Color ECG:", bg=self.T["panel"],
                 fg=self.T["muted"], font=("Segoe UI", 7, "bold")).pack(side=tk.LEFT)

        for c in self._ecg_color_presets:
            tk.Button(
                color_row, bg=c, width=2, height=1, relief="flat", bd=1,
                cursor="hand2",
                command=lambda col=c: self._set_ecg_color(col),
            ).pack(side=tk.LEFT, padx=1)

        # Etiqueta de sesion
        tk.Frame(col4, bg=self.T["border"], height=1).pack(fill=tk.X, pady=3)
        tk.Label(col4, text="Etiqueta de sesión:", bg=self.T["panel"],
                 fg=self.T["muted"], font=("Segoe UI", 7, "bold")).pack(anchor="w")
        sess_entry = tk.Entry(
            col4, textvariable=self.session_label_var,
            bg=self.T["neutral_bg"], fg=self.T["text"],
            relief="flat", font=("Segoe UI", 8), width=18,
            highlightthickness=1, highlightbackground=self.T["border"],
            insertbackground=self.T["text"],
        )
        sess_entry.pack(anchor="w", pady=(2, 0))

    def _build_tab_connection(self, parent):
        """Pestana CONEXION: puerto serial, baud rate, conectar/desconectar."""
        # Col 1: modo actual
        col1 = self._tab_col(parent, "MODO ACTUAL")

        self.hw_mode_badge = tk.Label(
            col1, text="MODO SIMULACIÓN", padx=10, pady=7, bd=0,
            font=("Segoe UI", 10, "bold"),
        )
        self.hw_mode_badge.pack(fill=tk.X, pady=(0, 6))
        self._set_badge(self.hw_mode_badge, "MODO SIMULACIÓN", "warning",
                        font=("Segoe UI", 10, "bold"))

        # Col 2: configuracion de puerto
        col2 = self._tab_col(parent, "PUERTO SERIAL")

        self._available_ports = list_available_ports() or [config.SERIAL_PORT]
        r_port = self._row(col2, "Puerto COM")
        self._port_menu = tk.OptionMenu(r_port, self.port_var, *self._available_ports)
        self._port_menu.configure(
            bg=self.T["neutral_bg"], fg=self.T["text"],
            activebackground=self.T["primary"], activeforeground="#FFFFFF",
            highlightthickness=1, highlightbackground=self.T["border"],
            relief="flat", font=("Segoe UI", 9), width=10,
        )
        self._port_menu["menu"].configure(
            bg=self.T["neutral_bg"], fg=self.T["text"],
            activebackground=self.T["primary"], activeforeground="#FFFFFF",
        )
        self._port_menu.pack()

        r_baud = self._row(col2, "Baud rate")
        baud_options = ["9600", "19200", "57600", "115200", "230400"]
        baud_menu = tk.OptionMenu(r_baud, self.baud_var, *baud_options)
        baud_menu.configure(
            bg=self.T["neutral_bg"], fg=self.T["text"],
            activebackground=self.T["primary"], activeforeground="#FFFFFF",
            highlightthickness=1, highlightbackground=self.T["border"],
            relief="flat", font=("Segoe UI", 9), width=10,
        )
        baud_menu["menu"].configure(
            bg=self.T["neutral_bg"], fg=self.T["text"],
            activebackground=self.T["primary"], activeforeground="#FFFFFF",
        )
        baud_menu.pack()

        # Col 3: botones de accion
        col3 = self._tab_col(parent, "ACCIONES")

        self.connect_btn = self._btn(
            col3, "Conectar", self.on_connect, kind="primary", pady=8,
        )
        self.connect_btn.pack(fill=tk.X, pady=(0, 4))

        self._btn(
            col3, "Actualizar puertos", self.on_refresh_ports,
            kind="neutral", pady=7,
        ).pack(fill=tk.X)

        # Col 4: estado de conexion
        col4 = self._tab_col(parent, "ESTADO DEL HARDWARE", sep=False)

        self.conn_esp32_badge   = self._metric_row(col4, "Estado ESP32")
        self.conn_samples_badge = self._metric_row(col4, "Muestras totales")

    def _build_tab_simulation(self, parent):
        """Pestana SIMULACION: parametros del generador de ECG sintetico."""
        # Col 1: frecuencia y amplitud
        col1 = self._tab_col(parent, "GENERADOR ECG")

        params = [
            ("Frec. cardiaca (BPM)", "Frecuencia simulada",
             self.sim_hr_var,    30.0, 200.0, 1.0),
            ("Amplitud del ECG",     "Multiplicador de amplitud R",
             self.sim_amp_var,   0.1,  3.0,   0.1),
            ("Nivel de ruido (mV)",  "Desviacion estandar del ruido",
             self.sim_noise_var, 0.0,  0.5,   0.01),
        ]

        for lbl, hlp, var, fr, to, inc in params:
            r = self._row(col1, lbl, hlp)
            self._spinbox(r, var, fr, to, inc, 7).pack()

        # Col 2: tipo de forma de onda y arritmia
        col2 = self._tab_col(parent, "CONDICIÓN CARDIACA", sep=False)

        r_wf = self._row(col2, "Tipo de onda")
        wf_options = ["ECG NORMAL", "BRADICARDIA", "TAQUICARDIA"]
        wf_menu = tk.OptionMenu(r_wf, self.sim_wf_var, *wf_options,
                                command=self._on_waveform_type_change)
        wf_menu.configure(
            bg=self.T["neutral_bg"], fg=self.T["text"],
            activebackground=self.T["primary"], activeforeground="#FFFFFF",
            highlightthickness=1, highlightbackground=self.T["border"],
            relief="flat", font=("Segoe UI", 9), width=14,
        )
        wf_menu["menu"].configure(
            bg=self.T["neutral_bg"], fg=self.T["text"],
            activebackground=self.T["primary"], activeforeground="#FFFFFF",
        )
        wf_menu.pack()

        self._btn(
            col2, "Agregar arritmia (5 s)", self.on_add_arrhythmia,
            kind="warning", pady=8,
        ).pack(fill=tk.X, pady=(8, 4))

        self.sim_status_badge = self._metric_row(col2, "Estado generador")
        self._set_badge(self.sim_status_badge, "EN EJECUCIÓN", "success")

    def _build_tab_hardware(self, parent):
        """Pestaña HARDWARE: referencia de pines ESP32 y tabla de selección de derivaciones."""

        def _pin_row(container, label, value, kind="info"):
            bg = self.T["info_bg"]    if kind == "info"    else self.T["warning_bg"]
            fg = self.T["info_fg"]    if kind == "info"    else self.T["warning_fg"]
            row = tk.Frame(container, bg=self.T["panel"])
            row.pack(fill=tk.X, pady=2)
            tk.Label(row, text=label, bg=self.T["panel"], fg=self.T["text"],
                     font=("Segoe UI", 8), anchor="w").pack(side=tk.LEFT, fill=tk.X, expand=True)
            tk.Label(row, text=value, bg=bg, fg=fg,
                     font=("Segoe UI", 8, "bold"), padx=6, pady=2).pack(side=tk.RIGHT)

        # ── Columna 1: Entradas al ESP32 ─────────────────────────
        col1 = self._tab_col(parent, "ENTRADAS — ESP32")
        _pin_row(col1, "ECG analógico",     "GPIO 36  (ADC1_CH0)")
        _pin_row(col1, "MUX — línea A",     "GPIO 26")
        _pin_row(col1, "MUX — línea B",     "GPIO 27")
        _pin_row(col1, "MUX — línea C",     "GPIO 14")
        _pin_row(col1, "Referencia VCC",    "3V3  (3.3 V)")
        _pin_row(col1, "Tierra",            "GND")

        # ── Columna 2: Salidas del ESP32 ──────────────────────────
        col2 = self._tab_col(parent, "SALIDAS — ESP32")
        _pin_row(col2, "Marcapasos  fase +",  "GPIO 25  (DAC 1)", kind="warn")
        _pin_row(col2, "Marcapasos  fase −",  "GPIO 26  (DAC 2)", kind="warn")
        _pin_row(col2, "Comunicación TX→PC",  "GPIO  1  (UART0 TX)", kind="warn")
        _pin_row(col2, "Comunicación RX←PC",  "GPIO  3  (UART0 RX)", kind="warn")
        _pin_row(col2, "Velocidad UART",      "115 200 baud", kind="warn")

        # ── Columna 3: Tabla de verdad MUX CD4051 ─────────────────
        col3 = self._tab_col(parent, "MUX CD4051 — SELECCIÓN")

        # Encabezado de tabla
        hdr = tk.Frame(col3, bg=self.T["neutral_bg"])
        hdr.pack(fill=tk.X, pady=(0, 3))
        for txt, w in [("A", 3), ("B", 3), ("C", 3), ("Derivación", 10)]:
            tk.Label(hdr, text=txt, bg=self.T["neutral_bg"], fg=self.T["neutral_fg"],
                     font=("Segoe UI", 7, "bold"), width=w, anchor="center").pack(side=tk.LEFT)

        _MUX_TABLE = [
            ("0", "0", "0", "I"),
            ("1", "0", "0", "II"),
            ("0", "1", "0", "III"),
            ("1", "1", "0", "aVR"),
            ("0", "0", "1", "aVL"),
            ("1", "0", "1", "aVF"),
        ]
        for a, b, c, lead in _MUX_TABLE:
            r = tk.Frame(col3, bg=self.T["panel"])
            r.pack(fill=tk.X, pady=1)
            for val, w in [(a, 3), (b, 3), (c, 3)]:
                tk.Label(r, text=val, bg=self.T["neutral_bg"], fg=self.T["text"],
                         font=("Segoe UI", 8), width=w, anchor="center").pack(side=tk.LEFT)
            tk.Label(r, text=lead, bg=self.T["info_bg"], fg=self.T["info_fg"],
                     font=("Segoe UI", 8, "bold"), padx=8, anchor="center").pack(side=tk.LEFT, padx=(4, 0))

        # ── Columna 4: Pasos para conectar hardware ───────────────
        col4 = self._tab_col(parent, "CÓMO CONECTAR EL HARDWARE", sep=False, min_width=210)

        pasos = [
            ("1", "Conectar el ESP32 al PC por USB."),
            ("2", "Clic en  ↻  para actualizar la lista de puertos."),
            ("3", "Seleccionar el puerto COM del ESP32 en el menú."),
            ("4", "Presionar  ⚡ CONECTAR  en la barra superior."),
            ("5", "El badge cambia a  HARDWARE  cuando hay señal."),
            ("6", "Para volver a simulación: presionar  ✖ DESCONECTAR."),
        ]
        for num, texto in pasos:
            fila = tk.Frame(col4, bg=self.T["panel"])
            fila.pack(fill=tk.X, pady=3)
            tk.Label(fila, text=num, bg=self.T["primary"], fg="white",
                     font=("Segoe UI", 8, "bold"), width=2, anchor="center",
                     padx=4, pady=2).pack(side=tk.LEFT)
            tk.Label(fila, text=texto, bg=self.T["panel"], fg=self.T["text"],
                     font=("Segoe UI", 7), wraplength=165, justify="left",
                     anchor="w").pack(side=tk.LEFT, padx=(6, 0), fill=tk.X, expand=True)

    # ----------------------------------------------------------
    def _create_status_bar(self, parent):
        """
        Barra de estado inferior (navy, 26px).
        Misma estetica que el topbar — completamente distinto al status bar gris original.
        """
        bar = tk.Frame(parent, bg=self.T["topbar_bg"], height=26)
        bar.pack(fill=tk.X, pady=(5, 0))
        bar.pack_propagate(False)

        def _sl(text, fg=None, bold=False):
            return tk.Label(
                bar, text=text,
                bg=self.T["topbar_bg"],
                fg=fg or self.T["topbar_muted"],
                font=("Segoe UI", 8, "bold" if bold else "normal"),
            )

        _sl("  Muestras:").pack(side=tk.LEFT)
        self.sb_samples_lbl = _sl("0", fg=self.T["topbar_text"], bold=True)
        self.sb_samples_lbl.pack(side=tk.LEFT, padx=(3, 0))

        tk.Frame(bar, bg=self.T["topbar_sep"], width=1).pack(
            side=tk.LEFT, fill=tk.Y, padx=12, pady=4
        )

        _sl("BPM:").pack(side=tk.LEFT)
        self.sb_bpm_lbl = tk.Label(
            bar, text="---",
            bg=self.T["topbar_bg"], fg=self.T["success"],
            font=("Segoe UI", 8, "bold"),
        )
        self.sb_bpm_lbl.pack(side=tk.LEFT, padx=(4, 0))

        tk.Frame(bar, bg=self.T["topbar_sep"], width=1).pack(
            side=tk.LEFT, fill=tk.Y, padx=12, pady=4
        )

        _sl("Ritmo:").pack(side=tk.LEFT)
        self.sb_rhythm_lbl = tk.Label(
            bar, text="---",
            bg=self.T["topbar_bg"], fg=self.T["topbar_muted"],
            font=("Segoe UI", 8, "bold"),
        )
        self.sb_rhythm_lbl.pack(side=tk.LEFT, padx=(4, 0))

        self.sb_mode_lbl = tk.Label(
            bar, text="MODO SIMULACIÓN",
            bg=self.T["topbar_bg"], fg=self.T["warning"],
            font=("Segoe UI", 8, "bold"),
        )
        self.sb_mode_lbl.pack(side=tk.RIGHT, padx=(0, 12))

        _sl("Frec.: 1000 Hz  |  Sesión:").pack(side=tk.RIGHT, padx=(0, 4))
        self.sb_session_lbl = tk.Label(
            bar, textvariable=self.session_label_var,
            bg=self.T["topbar_bg"], fg=self.T["topbar_text"],
            font=("Segoe UI", 8, "bold"),
        )
        self.sb_session_lbl.pack(side=tk.RIGHT, padx=(0, 6))

    # ==============================================================
    # ── HELPERS DE WIDGETS (estilo Clinical Light) ─────────────────
    # ==============================================================

    def _tab_col(self, parent, title=None, sep=True, min_width=160):
        """Columna vertical dentro del panel de pestañas."""
        col = tk.Frame(parent, bg=self.T["panel"], padx=12, pady=8)
        col.pack(side=tk.LEFT, fill=tk.BOTH)

        if min_width:
            col.configure(width=min_width)
            col.pack_propagate(False)

        if title:
            tk.Label(
                col, text=title,
                bg=self.T["panel"], fg=self.T["muted"],
                font=("Segoe UI", 7, "bold"),
            ).pack(anchor="w", pady=(0, 6))

        if sep:
            tk.Frame(parent, bg=self.T["border"], width=1).pack(
                side=tk.LEFT, fill=tk.Y, pady=6
            )

        return col

    def _set_badge(self, widget, text, kind="neutral", font=None):
        """Aplica color de badge a un Label segun el tipo."""
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
        """Boton de accion con estilo Clinical Light."""
        colors = {
            "primary": (self.T["primary"],  self.T["primary_active"]),
            "accent":  (self.T["accent"],   self.T["accent_active"]),
            "danger":  (self.T["danger"],   self.T["danger_active"]),
            "success": (self.T["success"],  "#047857"),
            "neutral": (self.T["neutral_bg"], "#CBD5E6"),
            "warning": (self.T["warning"],  "#B45309"),
        }
        bg, abg = colors.get(kind, colors["primary"])
        fg = self.T["text"] if kind == "neutral" else self.T["button_text"]
        afg = self.T["text"] if kind == "neutral" else self.T["button_text"]
        return tk.Button(
            parent, text=text, command=command,
            bg=bg, fg=fg,
            activebackground=abg, activeforeground=afg,
            relief="flat", bd=0, cursor="hand2",
            font=font or ("Segoe UI", 9, "bold"),
            padx=padx, pady=pady, highlightthickness=0,
        )

    def _spinbox(self, parent, var, from_, to, increment, width=10):
        """Spinbox con estilo Clinical Light (fondo claro, borde sutil)."""
        return tk.Spinbox(
            parent, from_=from_, to=to, increment=increment,
            textvariable=var, width=width,
            font=("Segoe UI", 9), justify="center",
            bg=self.T["neutral_bg"], fg=self.T["text"],
            relief="flat", bd=0,
            buttonbackground=self.T["border"],
            highlightthickness=1, highlightbackground=self.T["border"],
            highlightcolor=self.T["primary"],
            insertbackground=self.T["text"],
        )

    def _row(self, parent, label_text, help_text=None):
        """Fila etiqueta (izquierda) + widget (derecha) dentro de una columna de pestañas."""
        row = tk.Frame(parent, bg=self.T["panel"])
        row.pack(fill=tk.X, pady=5)

        left = tk.Frame(row, bg=self.T["panel"])
        left.pack(side=tk.LEFT, fill=tk.X, expand=True)

        tk.Label(
            left, text=label_text,
            bg=self.T["panel"], fg=self.T["text"],
            font=("Segoe UI", 8, "bold"),
        ).pack(anchor="w")

        if help_text:
            tk.Label(
                left, text=help_text,
                bg=self.T["panel"], fg=self.T["muted"],
                font=("Segoe UI", 6), wraplength=120, justify="left",
            ).pack(anchor="w")

        right = tk.Frame(row, bg=self.T["panel"])
        right.pack(side=tk.RIGHT, anchor="e")
        return right

    def _metric_row(self, parent, label_text):
        """Fila de metrica: etiqueta + badge de valor alineado a la derecha."""
        row = tk.Frame(parent, bg=self.T["panel"])
        row.pack(fill=tk.X, pady=2)

        tk.Label(
            row, text=label_text,
            bg=self.T["panel"], fg=self.T["muted"],
            font=("Segoe UI", 8),
        ).pack(side=tk.LEFT)

        badge = tk.Label(row, text="---", padx=8, pady=3, bd=0)
        badge.pack(side=tk.RIGHT)
        self._set_badge(badge, "---", "neutral")
        return badge

    # ==============================================================
    # ── PREVIEW BIFASICO ──────────────────────────────────────────
    # ==============================================================

    def _draw_biphasic_preview(self):
        """Dibuja la forma de onda bifasica en el canvas de preview (colores claros)."""
        canvas = self.pace_canvas
        canvas.update_idletasks()
        W = canvas.winfo_width() or 210
        H = 62

        canvas.delete("all")

        # Colores actualizados: ambar positivo, violeta negativo
        wave_pos = "#D97706"   # ambar — fase positiva
        wave_neg = "#7C3AED"   # violeta — fase negativa (vs rojo original)
        base_col = self.T["baseline"]

        cy  = H // 2
        amp = int(H * 0.33)
        m   = int(W * 0.08)

        x0 = m
        x1 = m + int((W - 2 * m) * 0.25)
        x2 = m + int((W - 2 * m) * 0.50)
        x3 = m + int((W - 2 * m) * 0.75)
        x4 = W - m

        # Baseline punteada
        canvas.create_line(0, cy, W, cy, fill=base_col, width=1, dash=(3, 3))

        # Etiquetas de fase
        canvas.create_text(int((x1 + x2) / 2), cy - amp - 8,
                           text="+", fill=wave_pos, font=("Segoe UI", 9, "bold"))
        canvas.create_text(int((x2 + x3) / 2), cy + amp + 8,
                           text="-", fill=wave_neg, font=("Segoe UI", 9, "bold"))

        # Fase positiva
        canvas.create_line(
            x0, cy, x1, cy, x1, cy - amp, x2, cy - amp,
            fill=wave_pos, width=2, smooth=False,
        )
        # Fase negativa
        canvas.create_line(
            x2, cy + amp, x3, cy + amp, x3, cy, x4, cy,
            fill=wave_neg, width=2, smooth=False,
        )
        # Transicion instantanea entre fases
        canvas.create_line(x2, cy - amp, x2, cy + amp,
                           fill=self.T["muted"], width=1)

        canvas.create_text(x0, H - 4, text="0",
                           fill=self.T["muted"], font=("Segoe UI", 7))
        canvas.create_text(x4, H - 4, text="T",
                           fill=self.T["muted"], font=("Segoe UI", 7))

    # ==============================================================
    # ── RELOJ ─────────────────────────────────────────────────────
    # ==============================================================

    def _update_clock(self):
        if not self.is_running:
            return
        self.clock_label.config(text=time.strftime("%H:%M:%S"))
        self.after(1000, self._update_clock)

    # ==============================================================
    # ── ACTUALIZADOR DE BOTONES DE DERIVACION (adaptado al topbar) ─
    # ==============================================================

    def _update_lead_buttons(self):
        """Resalta el boton de derivada activa en el topbar (colores navy)."""
        active = self.app_state.current_mux_state
        label  = self.app_state.mux_state_label.get(active, "?")

        for state, btn in self._lead_buttons.items():
            if state == active:
                btn.config(bg=self.T["primary"], fg="#FFFFFF")
            else:
                btn.config(bg=self.T["topbar_btn"], fg=self.T["topbar_text"])

        if hasattr(self, "active_lead_badge"):
            self._set_badge(self.active_lead_badge, f"DERIVACIÓN {label}", "info",
                            font=("Segoe UI", 13, "bold"))

        if hasattr(self, "lead_title_label"):
            self.lead_title_label.config(text=f"Derivación {label}")

        if hasattr(self, "ax"):
            self.ax.set_title("")   # título se muestra en la barra tkinter superior

    # ==============================================================
    # ── HELPERS DE SEÑAL ECG (sin cambios) ────────────────────────
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
        Retorna: (y_out, x_start_idx, x_end_idx)
        """
        y_out       = np.array(y_signal, dtype=float).copy()
        sample_rate = int(getattr(config, "SAMPLE_RATE", 1000))
        total_s     = max(2, int(round(sample_rate * duration_ms / 1000.0)))
        half_s      = total_s // 2

        if len(y_out) < total_s + 4:
            return y_out, 0, 0

        end_idx   = len(y_out) - 2
        start_idx = max(0, end_idx - total_s)

        ref_start  = max(0, start_idx - int(sample_rate * 0.05))
        local_base = float(np.median(y_out[ref_start:start_idx])) if start_idx > ref_start else 0.0

        y_out[start_idx:start_idx + half_s] = local_base + amplitude
        y_out[start_idx + half_s:end_idx]   = local_base - amplitude
        if start_idx > 0:
            y_out[start_idx - 1] = local_base
        if end_idx < len(y_out):
            y_out[end_idx] = local_base

        return y_out, start_idx, end_idx

    # ==============================================================
    # ── ACTUALIZADORES DE PANELES (sin cambios de logica) ─────────
    # ==============================================================

    def _update_vital_signs(self, bpm: float, rhythm: str, qrs_count: int, sig_ok: bool):
        """Actualiza la tira de metricas con los valores del frame actual."""
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

        rhythm_upper = (rhythm or "").upper()
        rhythm_ui_map = {
            "NORMAL":      "NORMAL",
            "BRADYCARDIA": "BRADICARDIA",
            "TACHYCARDIA": "TAQUICARDIA",
            "ASYSTOLE":    "ASISTOLIA",
        }
        rhythm_colors = {
            "NORMAL":      "success",
            "BRADYCARDIA": "warning",
            "TACHYCARDIA": "warning",
            "ASYSTOLE":    "danger",
        }
        kind = rhythm_colors.get(rhythm_upper, "neutral")
        self._set_badge(self.rhythm_badge,
                        rhythm_ui_map.get(rhythm_upper, rhythm_upper or "---"),
                        kind, font=("Segoe UI", 11, "bold"))

        self._set_badge(self.qrs_count_badge, str(qrs_count), "info")

        if not sig_ok:
            self._set_badge(self.sig_quality_badge, "SIN SEÑAL", "neutral")
        elif bpm > 0:
            self._set_badge(self.sig_quality_badge, "BUENA", "success")
        else:
            self._set_badge(self.sig_quality_badge, "BAJA", "warning")

        # Intervalo R-R
        if self._last_rr_ms > 0:
            rr_color = "accent" if 600 <= self._last_rr_ms <= 1000 else "warning"
            self.rr_interval_badge.config(text=f"{self._last_rr_ms:.0f}")
            self._set_badge(self.rr_interval_badge, f"{self._last_rr_ms:.0f}", rr_color)
        else:
            self._set_badge(self.rr_interval_badge, "---", "neutral")

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
                        font=("Segoe UI", 10, "bold"))

        amp = self._safe_float(self.app_state.pace_amplitude_var, 1.0)
        dur = self._safe_float(self.pace_duration_ms_var, 4.0)
        bpm = self._safe_float(self.app_state.pace_bpm_var, 60.0)

        self._set_badge(self.pace_amp_badge,  f"{amp:.1f} V",   "danger")
        self._set_badge(self.pace_dur_badge,  f"{dur:.0f} ms",  "info")
        self._set_badge(self.pace_bpm_badge,  f"{bpm:.0f} BPM", "accent")

        # Energia del pulso: E = V^2 * t / R (R_tejido ≈ 500 Ω)
        energy_uj = (amp ** 2) * (dur / 1000.0) / 500.0 * 1e6
        if hasattr(self, "pace_energy_badge"):
            self._set_badge(self.pace_energy_badge, f"{energy_uj:.1f} µJ", "warning")

    def _update_connection_panel(self):
        """Actualiza badges de conexion segun el estado actual."""
        sim  = getattr(self.app_state, "simulation_mode", True)
        conn = self.app_state.esp32_connected

        if conn:
            self._set_badge(self.hw_mode_badge,   "MODO HARDWARE",  "success",
                            font=("Segoe UI", 10, "bold"))
            self._set_badge(self.conn_esp32_badge, "EN LÍNEA",       "success")
            self.connect_btn.config(text="Desconectar", bg=self.T["danger"])
            self._set_badge(self.mode_badge_hdr,  "HARDWARE",        "success")
            # Topbar
            if hasattr(self, "topbar_hw_btn"):
                self.topbar_hw_btn.config(
                    text="✖ DESCONECTAR", bg=self.T["danger"],
                    activebackground=self.T["danger_active"],
                )
        else:
            lbl  = "MODO SIMULACIÓN" if sim else "DESCONECTADO"
            kind = "warning"          if sim else "danger"
            self._set_badge(self.hw_mode_badge,   lbl,  kind,
                            font=("Segoe UI", 10, "bold"))
            self._set_badge(self.mode_badge_hdr,
                            "SIMULACIÓN" if sim else "DESCONECTADO", kind)
            self._set_badge(self.conn_esp32_badge, "DESCONECTADO", "danger")
            self.connect_btn.config(text="Conectar", bg=self.T["primary"])
            # Topbar
            if hasattr(self, "topbar_hw_btn"):
                self.topbar_hw_btn.config(
                    text="⚡ CONECTAR", bg="#D97706",
                    activebackground="#B45309",
                )

        self._set_badge(self.conn_samples_badge,
                        f"{self.app_state.sample_count:,}", "neutral")

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
        self.serial_reader.pace_bpm = self._safe_float(
            self.app_state.pace_bpm_var, 60.0
        )

        arrhythmia = getattr(self.serial_reader, "sim_arrhythmia", False)
        if arrhythmia:
            self._set_badge(self.sim_status_badge, "ARRITMIA", "warning")
        else:
            self._set_badge(self.sim_status_badge, "EN EJECUCIÓN", "success")

    # ==============================================================
    # ── ACCIONES DE USUARIO (sin cambios de logica) ───────────────
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

    def on_pace_trigger(self):
        """Activa el trigger manual del marcapasos."""
        now = time.time()
        self.app_state.pace_pulse_pending = True
        hold = max(0.5, self._safe_float(self.pace_alert_hold_var, 1.5))
        self.app_state.pace_alert_until = now + hold

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

    def on_connect(self):
        """Conecta al puerto seleccionado o desconecta el hardware."""
        if self.app_state.esp32_connected:
            self._restart_reader(port="NONE_DISCONNECT")
        else:
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
    # ── BUCLE PRINCIPAL DE ACTUALIZACION (sin cambios) ────────────
    # ==============================================================

    def update_gui(self):
        """
        Bucle principal de refresco de la GUI.
        RAPIDA (~80ms): grafico matplotlib, deteccion de picos.
        LENTA  (~400ms): badges de metricas y pestanas.
        """
        if not self.is_running:
            return

        try:
            self._update_gui_impl()
        except Exception:
            try:
                import traceback
                traceback.print_exc()
            except Exception:
                pass
        finally:
            refresh = max(40, self._safe_int(self.refresh_interval_var, 80))
            self.after(refresh, self.update_gui)

    def _update_gui_impl(self):
        """Implementacion del refresco GUI (se llama desde update_gui)."""
        now     = time.time()

        self._frame_count += 1
        do_slow = (self._frame_count % 5 == 0)

        win     = max(100, self._safe_int(self.app_state.window_size, 2000))
        gain    = max(0.1,  self._safe_float(self.app_state.ecg_gain, 1.0))
        y_max_v = max(0.3,  self._safe_float(self.app_state.y_max, 2.0))
        in_blank = now < getattr(self.app_state, "blank_until", 0.0)

        with self.app_state.data_lock:
            x_buf = list(self.app_state.time_buffer)
            y_buf = list(self.app_state.voltage_buffer)
            sc    = int(self.app_state.sample_count)

        if len(x_buf) > 1:
            xw_raw = x_buf[-win:]
            yw     = np.array(y_buf[-win:], dtype=float)
        else:
            xw_raw = list(range(max(2, win)))
            yw     = np.zeros(max(2, win), dtype=float)

        sample_rate = float(getattr(config, "SAMPLE_RATE", 1000))
        xw_sec      = np.asarray(xw_raw, dtype=float) / sample_rate

        signal_ok = (not in_blank) and self._signal_present(yw)
        self._update_no_signal_state(signal_ok)
        no_sig = getattr(self.app_state, "no_signal", False)

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

        n_dc       = min(len(yw), 500)
        dc_offset  = float(np.median(yw[-n_dc:]))
        y_centered = (yw - dc_offset) * gain

        try:
            while True:
                peaks, qrs, bpm, rhythm = self._analysis_out_q.get_nowait()
                self._analysis_peaks  = peaks or []
                self._analysis_qrs    = qrs or []
                self._analysis_bpm    = float(bpm or 0.0)
                self._analysis_rhythm = rhythm or "---"
        except Exception:
            pass

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

        # Calcular intervalo R-R
        if len(peaks) >= 2:
            rr_arr = np.diff(peaks)
            if len(rr_arr) > 0:
                self._last_rr_ms = float(np.median(rr_arr)) / sample_rate * 1000.0
        else:
            self._last_rr_ms = 0.0

        for pi in peaks:
            if pi < len(xw_raw):
                abs_idx = int(xw_raw[pi])
                if abs_idx > self._last_qrs_abs_idx:
                    self.app_state.qrs_detected_count += 1
                    self._last_qrs_abs_idx = abs_idx
        self._last_qrs_complexes = list(self._analysis_qrs or [])

        # Autoscala Y si está activa
        if self.autoscale_y_var.get() and len(y_centered) > 0:
            _p = float(np.percentile(np.abs(y_centered), 98))
            y_max_v = max(0.3, _p * 1.25)

        self.ecg_line.set_data(xw_sec, y_centered)
        self.ax.set_xlim(xw_sec[0], xw_sec[-1])
        self.ax.set_ylim(-y_max_v, y_max_v)

        # Aplicar toggles de visualizacion
        self.baseline_line.set_visible(self.show_baseline_var.get())

        show_pk = self.show_peaks_var.get()
        if peaks and show_pk:
            self.peaks_line.set_data(
                [xw_sec[i] for i in peaks if i < len(xw_sec)],
                [y_centered[i] for i in peaks if i < len(y_centered)],
            )
            self.peaks_line.set_visible(True)
        else:
            self.peaks_line.set_data([], [])
            self.peaks_line.set_visible(show_pk)

        show_qrs = self.show_qrs_var.get()
        if show_qrs:
            y_qrs = np.full(len(y_centered), np.nan)
            for qrs in self._last_qrs_complexes:
                o, f = qrs["onset"], qrs["offset"]
                y_qrs[o:f + 1] = y_centered[o:f + 1]
            self.qrs_line.set_data(xw_sec, y_qrs)
            self.qrs_line.set_visible(True)
        else:
            self.qrs_line.set_data([], [])
            self.qrs_line.set_visible(False)

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

        # Cuadricula: solo actualizar en frames lentos para no degradar rendimiento
        if do_slow:
            self.ax.grid(self.show_grid_var.get(),
                         color=self.T["grid"], alpha=0.9, linewidth=0.6,
                         linestyle="-", which="both")

        self.mpl_canvas.draw_idle()

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

    # ==============================================================
    # ── CONTROL DE MODO AUTO (sin cambios) ────────────────────────
    # ==============================================================

    def check_auto_mode(self):
        """Gestiona el cambio automatico de derivadas en modo AUTO."""
        if not self.is_running:
            return

        now = time.time()

        if not self.auto_scan_active:
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
    # ── CIERRE DE LA APLICACION (sin cambios) ─────────────────────
    # ==============================================================

    def on_closing(self):
        """Cierra la aplicacion de forma limpia."""
        self.is_running        = False
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
