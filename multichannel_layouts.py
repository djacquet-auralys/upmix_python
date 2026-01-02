# -*- coding: utf-8 -*-
"""
Created on Wed May 14 14:55:50 2025

@author: Damien

spk_layouts.py  –  petits utilitaires pour disposer automatiquement
                  les haut-parleurs 2D de layouts courants.
"""
from __future__ import annotations

from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------------
# 1) Définition des layouts : labels et azimuts par défaut (° ITU/Dolby)
# ---------------------------------------------------------------------
_LAYOUTS: Dict[str, Dict[str, List]] = {
    "stereo": {
        "labels": ["L", "R"],
        "az": [-30, 30],
    },
    "2.1": {
        "labels": ["L", "R", "LFE"],
        "az": [-30, 30, 0],
    },
    "5.0": {
        "labels": ["L", "R", "C", "LS", "RS"],
        "az": [-30, 30, 0, -110, 110],
    },
    "5.1": {
        "labels": ["L", "R", "C", "LFE", "LS", "RS"],
        "az": [-30, 30, 0, 0, -110, 110],
    },
    "6.1": {
        "labels": ["L", "R", "C", "LS", "RS", "Cs", "LFE"],
        "az": [-30, 30, 0, -110, 110, 180, 0],
    },
    "7.1": {
        "labels": ["L", "R", "C", "LFE", "LS", "RS", "LB", "RB"],
        "az": [-30, 30, 0, 0, -90, 90, -150, 150],
    },
    "8.0": {
        "labels": [f"S{i+1}" for i in range(8)],
        "az": [i * 45 for i in range(8)],
    },
    "8.1": {
        "labels": ["L", "R", "C", "LS", "RS", "LB", "RB", "Cs", "LFE"],
        "az": [-30, 30, 0, -90, 90, -150, 150, 180, 0],
    },
    "10.2": {
        # 10 canaux full-range uniformément répartis + 2 LFE
        "labels": [f"S{i+1}" for i in range(10)] + ["LFE1", "LFE2"],
        "az": [i * 36 for i in range(10)] + [0, 0],
    },
    "11.1": {
        # 1 centre + 10 autour + 1 LFE
        "labels": ["C"] + [f"S{i+1}" for i in range(10)] + ["LFE"],
        "az": [0] + [i * 36 for i in range(10)] + [0],
    },
    "13.1": {
        # 1 centre + 12 uniformes + 1 LFE
        "labels": ["C"] + [f"S{i+1}" for i in range(12)] + ["LFE"],
        "az": [0] + [i * 30 for i in range(12)] + [0],
    },
    "22.2": {
        # 18 canaux horizontaux + 2 LFE
        "labels": [f"H{i+1}" for i in range(18)] + ["LFE1", "LFE2"],
        "az": [i * 20 for i in range(18)] + [0, 0],
    },
    "quad": {
        "labels": ["L", "R", "LB", "RB"],
        "az": [-45, 45, -135, 135],
    },
    "sdds7.1": {
        "labels": ["L", "R", "Lc", "Rc", "C", "LS", "RS", "LFE"],
        "az": [-45, 45, -22.5, 22.5, 0, -100, 100, 0],
    },
    "auro9.1": {
        "labels": ["L", "R", "C", "LFE", "LW", "RW", "LS", "RS", "CS"],
        "az": [-30, 30, 0, 0, -60, 60, -110, 110, 180],
    },
}


# -----------------------------------------------------------------------------
# 2) Fonction publique
# -----------------------------------------------------------------------------
def get_spk_coordinates(
    layout: str,
    radius: float,
    *,
    az_offset: float = 0.0,
    center: Tuple[float, float] = (0.0, 0.0),
) -> Dict[str, Any]:
    """
    Parameters
    ----------
    layout : str
        Nom du layout ('stereo', '5.1', '7.1', …). Casse et points ignorés.
    radius : float
        Rayon (m) du cercle (ou demi-largeur pour un rectangle).
    az_offset : float, optional
        Décalage angulaire global appliqué à tous les haut-parleurs (°).
        Utile si l’avant n’est pas l’axe 0° de l’installation.
    center : tuple(float, float), optional
        Décalage XY du centre (permet de ne pas être à l’origine).

    Returns
    -------
    dict
        {
          'layout'          : '5.1',
          'labels'          : [...],
          'azimuth_deg'     : [...],
          'positions_xy'    : Nx2 np.ndarray,
          'orient_deg'      : [...],
          'radius'          : radius,
          'nb_spk'          : N_non_LFE,
          'nb_lfe'          : N_LFE
        }

    Notes
    -----
    - 0° par convention = +y (avant), angles positifs -> droite (sens horaire).
    - Les haut-parleurs sont orientés vers le point `center`.
    """
    name = layout.lower().replace(" ", "").replace("-", "").replace("_", "")
    if name not in _LAYOUTS:
        raise ValueError(f"Layout « {layout} » inconnu.")

    labels: List[str] = _LAYOUTS[name]["labels"]
    az_deg_base: List[float] = _LAYOUTS[name]["az"]

    # applique un offset éventuel
    az_deg = [(az + az_offset) % 360 for az in az_deg_base]

    # positions cartésiennes
    az_rad = np.deg2rad(az_deg)
    x_coords = radius * np.sin(az_rad) + center[0]
    y_coords = radius * np.cos(az_rad) + center[1]
    positions = np.column_stack((x_coords, y_coords))

    # orientations : chaque HP regarde le centre → az+180 (vers l’origine)
    orient_deg = [(az + 180) % 360 for az in az_deg]

    # compte LFE
    nb_lfe = sum(1 for lbl in labels if "LFE" in lbl.upper())
    nb_spk = len(labels) - nb_lfe

    return {
        "layout": layout,
        "labels": labels,
        "azimuth_deg": np.array(az_deg),
        "positions_xy": positions,  # numpy (N,2)
        "orient_deg": np.array(orient_deg),
        "radius": radius,
        "nb_spk": nb_spk,
        "nb_lfe": nb_lfe,
    }


# -----------------------------------------------------------------------------
# 3) Exemple d’usage
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    import pprint

    data = get_spk_coordinates("22.2", radius=2.0)
    pprint.pp(data)

    fig, ax = plt.subplots()
    positions_xy = data["positions_xy"]  # type: ignore
    labels = data["labels"]  # type: ignore
    ax.scatter(
        positions_xy[:, 0],  # ‹scatter› ou ‹plot› au choix
        positions_xy[:, 1],
        marker="^",
        s=80,
        color="tab:blue",
    )

    # ───── affichage des labels ─────
    for (x, y), label in zip(positions_xy, labels):
        ax.text(
            x,
            y + 0.08,
            label,  # petit décalage vertical
            ha="center",
            va="bottom",
            fontsize=9,
            weight="bold",
        )

    ax.set_aspect("equal")
    ax.grid(which="both", linestyle="--", alpha=0.4)
    ax.set_xlabel("x (m, droite +)")
    ax.set_ylabel("y (m, avant +)")
    plt.show()
