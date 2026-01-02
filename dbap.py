# -*- coding: utf-8 -*-
"""
Created on Thu May 15 12:10:41 2025

@author: Damien
"""
import numpy as np

def dbap_2d(spk_pos,
            source_pos,
            *,
            width: float = 0,
            blur: float = 1,
            rolloff: float = 6,
            constant_gain: bool = True,
            min_delays: bool = True):
    """
    Distance-Based Amplitude Panning (DBAP) pour un ensemble de haut-parleurs 2D.

    Parameters
    ----------
    spk_pos : array-like, shape (N, 2) ou (2N,)
        Coordonnées (x, y) des haut-parleurs, en mètres.
    source_pos : array-like, shape (2,)
        Coordonnées (x, y) de la source virtuelle, en mètres.
    width : float, default 0
        Largeur apparente de la source (en mètres).
    blur : float, default 1
        « Blur » DBAP (en mètres).
    rolloff : float, default 6
        Atténuation (dB) pour un doublement de distance.
    constant_gain : bool, default True
        Normalise les gains pour conserver une énergie totale constante.
    min_delays : bool, default True
        Décale tous les délais pour que le plus petit soit nul.

    Returns
    -------
    gains : ndarray, shape (N,)
        Gains linéaires à appliquer sur chaque haut-parleur.
    delays : ndarray, shape (N,)
        Retards de propagation (en secondes) pour chaque haut-parleur.
    """
    # --- Mise en forme & vérifications ------------------------------------------
    spk_pos = np.asarray(spk_pos, dtype=float)
    if spk_pos.ndim == 1:                      # liste plate -> reshape
        if spk_pos.size % 2:
            raise ValueError("Le tableau plat 'spk_pos' doit contenir un nombre pair de valeurs (x, y).")
        spk_pos = spk_pos.reshape(-1, 2)
    elif spk_pos.ndim == 2 and spk_pos.shape[1] != 2:
        raise ValueError("Chaque haut-parleur doit être décrit par deux coordonnées (x, y).")

    source_pos = np.asarray(source_pos, dtype=float).ravel()
    if source_pos.size != 2:
        raise ValueError("La position de la source doit comporter exactement deux valeurs (x, y).")

    # --- Constantes physiques & facteurs DBAP -----------------------------------
    C              = 343.0        # vitesse du son (m/s)
    BLUR_REF       = 1.0
    ROLLOFF_REF    = 6.0

    blur2          = blur ** 2
    width2         = (20 * width ** 2) ** 2   # traitement « historique » du paramètre width
    rolloff_ratio  = -rolloff / ROLLOFF_REF

    # --- Distances --------------------------------------------------------------
    diff      = spk_pos - source_pos
    sq_dist   = np.einsum('ij,ij->i', diff, diff)        # distance²
    dist      = np.sqrt(sq_dist)

    # --- Gains ------------------------------------------------------------------
    sq_gain   = (sq_dist + blur2 + width2) ** rolloff_ratio

    energy_ref  = np.sum(1.0 / (sq_dist + BLUR_REF))
    energy_real = np.sum(sq_gain)
    sq_gain    *= energy_ref / energy_real               # normalisation « référence »

    if constant_gain:
        sq_gain /= sq_gain.sum()

    gains  = np.sqrt(sq_gain)
    delays = dist / C

    if min_delays:
        delays -= delays.min()

    return gains, delays