# core/config.py

from __future__ import annotations

# Constantes physiques / pondérations (ajuste selon ton script original)
C_SOUND = 343.0          # m/s
MIN_GAIN = 1e-4          # sécurité numérique éventuelle

# Pondérations psychoacoustiques (si utilisées dans compute_seat_score)
A1_QDIR   = 0.4
A2_QSPACE = 0.3
A3_QWIDTH = 0.2
A4_QBLUR  = 0.1

# Autres constantes modèle (ex: JNDdir, CV0, sref/dref) à déplacer ici au besoin
SREF = 1.0
DREF = 1.0
JND_DIR = 5.0    # exemple
CV0 = 0.2        # exemple
