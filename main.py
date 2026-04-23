"""
main.py
Punto de entrada del Monitor ECG + Marcapasos.

Uso:
    python main.py

Si no hay hardware ESP32 conectado, la aplicacion arranca
automaticamente en modo SIMULACION con señal ECG matematica.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src import config
from src.appUI import ECGApp


def main():
    print("=" * 55)
    print("  Monitor ECG de Signos Vitales — Ingeniería Biomédica")
    print("=" * 55)
    print(f"  Puerto   : {config.SERIAL_PORT}")
    print(f"  Baudrate : {config.BAUDRATE}")
    print(f"  Muestreo : {config.SAMPLE_RATE} Hz")
    print(f"  Sesión   : {getattr(config, 'SESSION_LABEL', 'DEMO')}")
    print("=" * 55)
    print("  Iniciando interfaz... (cierre la ventana para salir)")
    print()

    try:
        app = ECGApp()
        app.mainloop()
    except KeyboardInterrupt:
        print("\n[Sistema] Interrupción por teclado. Cerrando.")
        sys.exit(0)


if __name__ == "__main__":
    main()
