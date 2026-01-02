# -*- coding: utf-8 -*-
"""
mask_generator.py - Génération et lissage des masques d'extraction

Ce module implémente :
1. LUT masque : création de la Look-Up Table pour l'extraction
2. Blur fréquentiel : lissage triangulaire sur l'axe des fréquences
3. Rampsmooth temporel : lissage attack/release sur l'axe temporel

@author: Damien
"""

from typing import Optional, Tuple

import numpy as np


def create_mask_lut(
    pan: float,
    width: float,
    slope: float,
    min_gain_db: float,
    resolution: int = 200,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Crée une Look-Up Table (LUT) pour le masque d'extraction.

    Parameters
    ----------
    pan : float
        Position centrale du masque, entre -1 et 1.
    width : float
        Largeur du masque (plage où le gain est maximal).
    slope : float
        Pente de la transition en dB par unité de pan.
    min_gain_db : float
        Gain minimum (floor) en dB (typiquement -40 dB).
    resolution : int
        Nombre de points dans la LUT (défaut: 200).

    Returns
    -------
    x : np.ndarray
        Positions de panning, shape (resolution,).
    lut : np.ndarray
        Gains linéaires correspondants, shape (resolution,).

    Notes
    -----
    Formule : y = 10^(max(min(SLOPE * (W/2 - abs(x - PAN)), 0), min_gain) / 20.0)

    La LUT a un pic à la position pan, décroît linéairement selon slope,
    et est limitée par min_gain.
    """
    # Créer l'axe x uniformément réparti entre -1 et 1
    x = np.linspace(-1.0, 1.0, resolution, dtype=np.float32)

    # Calcul du gain en dB selon la formule
    # gain_db = SLOPE * (W/2 - |x - PAN|)
    # Clippé entre min_gain et 0
    distance_from_pan = np.abs(x - pan)
    gain_db = slope * (width / 2.0 - distance_from_pan)
    gain_db = np.clip(gain_db, min_gain_db, 0.0)

    # Conversion en linéaire
    lut = np.power(10.0, gain_db / 20.0).astype(np.float32)

    return x, lut


def interpolate_lut(
    x_lut: np.ndarray,
    lut: np.ndarray,
    pan_values: np.ndarray,
) -> np.ndarray:
    """
    Interpole la LUT pour des valeurs de panning arbitraires.

    Parameters
    ----------
    x_lut : np.ndarray
        Positions de panning de la LUT, shape (resolution,).
    lut : np.ndarray
        Gains de la LUT, shape (resolution,).
    pan_values : np.ndarray
        Valeurs de panning pour lesquelles interpoler.
        Peut être de n'importe quelle forme.

    Returns
    -------
    gains : np.ndarray
        Gains interpolés, même forme que pan_values.
    """
    # np.interp fonctionne sur des tableaux 1D, donc on aplatit
    original_shape = pan_values.shape
    pan_flat = pan_values.flatten()

    # Interpolation linéaire
    gains_flat = np.interp(pan_flat, x_lut, lut)

    return gains_flat.reshape(original_shape).astype(np.float32)


def apply_freq_blur(
    gains: np.ndarray,
    exclude_dc_nyquist: bool = True,
) -> np.ndarray:
    """
    Applique un blur triangulaire sur l'axe fréquentiel.

    Parameters
    ----------
    gains : np.ndarray
        Gains à lisser.
        Shape: (n_freq,) ou (n_frames, n_freq).
    exclude_dc_nyquist : bool
        Si True, ne pas appliquer le blur aux bins 0 et Nyquist.

    Returns
    -------
    blurred : np.ndarray
        Gains lissés, même shape que l'entrée.

    Notes
    -----
    Le noyau triangulaire est [0.25, 0.5, 0.25] (normalisé).
    Le blur est appliqué sur ±1 bin (3 bins au total).
    """
    gains = np.asarray(gains, dtype=np.float32)

    # Noyau triangulaire normalisé
    kernel = np.array([0.25, 0.5, 0.25], dtype=np.float32)

    if gains.ndim == 1:
        # Cas 1D: (n_freq,)
        blurred = _apply_freq_blur_1d(gains, kernel, exclude_dc_nyquist)
    elif gains.ndim == 2:
        # Cas 2D: (n_frames, n_freq)
        blurred = np.zeros_like(gains)
        for i in range(gains.shape[0]):
            blurred[i] = _apply_freq_blur_1d(gains[i], kernel, exclude_dc_nyquist)
    else:
        raise ValueError(f"gains doit avoir 1 ou 2 dimensions, reçu {gains.ndim}")

    return blurred


def _apply_freq_blur_1d(
    gains: np.ndarray,
    kernel: np.ndarray,
    exclude_dc_nyquist: bool,
) -> np.ndarray:
    """Applique le blur fréquentiel à un vecteur 1D."""
    n_freq = len(gains)
    blurred = np.copy(gains)

    # Indices à traiter (excluant DC et Nyquist si demandé)
    start_idx = 1 if exclude_dc_nyquist else 0
    end_idx = n_freq - 1 if exclude_dc_nyquist else n_freq

    for i in range(start_idx, end_idx):
        if i > 0 and i < n_freq - 1:
            # Application du noyau [0.25, 0.5, 0.25]
            blurred[i] = (
                kernel[0] * gains[i - 1]
                + kernel[1] * gains[i]
                + kernel[2] * gains[i + 1]
            )

    return blurred.astype(np.float32)


class RampSmooth:
    """
    Lissage temporel avec attack et release (rampsmooth).

    Algorithme inspiré de gen~ RNBO (MaxMSP).
    """

    def __init__(
        self,
        n_freq: int,
        attack_frames: float = 1.0,
        release_frames: float = 50.0,
        freeze_threshold: float = 1e-6,
        double_release_dc_nyquist: bool = True,
    ):
        """
        Initialise le lisseur temporel.

        Parameters
        ----------
        n_freq : int
            Nombre de bins fréquentiels.
        attack_frames : float
            Temps d'attaque en frames STFT.
        release_frames : float
            Temps de release en frames STFT.
        freeze_threshold : float
            Seuil de puissance en dessous duquel le lissage est gelé.
        double_release_dc_nyquist : bool
            Si True, double le release pour les bins 0 et Nyquist.
        """
        self.n_freq = n_freq
        self.attack_frames = max(attack_frames, 1.0)  # Minimum 1 frame
        self.release_frames = max(release_frames, 1.0)
        self.freeze_threshold = freeze_threshold
        self.double_release_dc_nyquist = double_release_dc_nyquist

        # État interne: gains lissés
        self._state: Optional[np.ndarray] = None

    def reset(self) -> None:
        """Réinitialise l'état interne."""
        self._state = None

    def process(
        self,
        target_gains: np.ndarray,
        power: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Traite une frame de gains.

        Parameters
        ----------
        target_gains : np.ndarray
            Gains cibles pour cette frame, shape (n_freq,).
        power : np.ndarray, optional
            Puissance du signal pour le freeze (si < freeze_threshold).

        Returns
        -------
        smoothed : np.ndarray
            Gains lissés, shape (n_freq,).
        """
        target_gains = np.asarray(target_gains, dtype=np.float32)

        if self._state is None:
            # Première frame: initialiser avec les valeurs cibles
            initial_state = target_gains.copy()
            self._state = initial_state
            return initial_state.copy()

        # Identifier les bins à geler (puissance trop faible)
        freeze_mask = np.zeros(self.n_freq, dtype=bool)
        if power is not None:
            freeze_mask = power < self.freeze_threshold

        # Calculer les coefficients de lissage
        attack_coef = 1.0 / self.attack_frames
        release_coef = 1.0 / self.release_frames

        # Créer les coefficients par bin
        coefs = np.where(
            target_gains > self._state,
            attack_coef,  # Ramp up: utiliser attack
            release_coef,  # Ramp down: utiliser release
        )

        # Doubler le release pour DC et Nyquist si demandé
        if self.double_release_dc_nyquist:
            # Pour les bins 0 et Nyquist, release est doublé (coef divisé par 2)
            if target_gains[0] <= self._state[0]:
                coefs[0] = release_coef / 2.0
            if len(target_gains) > 1 and target_gains[-1] <= self._state[-1]:
                coefs[-1] = release_coef / 2.0

        # Ne pas mettre à jour si gelé (coef = 0)
        coefs[freeze_mask] = 0.0

        # Appliquer le lissage: state += (target - state) * coef
        delta = target_gains - self._state
        self._state = self._state + delta * coefs

        # Type assertion pour le type checker
        assert self._state is not None
        return self._state.copy()

    def process_batch(
        self,
        target_gains: np.ndarray,
        power: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Traite un batch de frames.

        Parameters
        ----------
        target_gains : np.ndarray
            Gains cibles, shape (n_frames, n_freq).
        power : np.ndarray, optional
            Puissance du signal, shape (n_frames, n_freq).

        Returns
        -------
        smoothed : np.ndarray
            Gains lissés, shape (n_frames, n_freq).
        """
        n_frames = target_gains.shape[0]
        smoothed = np.zeros_like(target_gains)

        for i in range(n_frames):
            pwr = power[i] if power is not None else None
            smoothed[i] = self.process(target_gains[i], pwr)

        return smoothed


def apply_temporal_smoothing(
    gains: np.ndarray,
    attack_frames: float,
    release_frames: float,
    freeze_threshold: float = 1e-6,
    power: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Applique le lissage temporel (rampsmooth) à un ensemble de gains.

    Parameters
    ----------
    gains : np.ndarray
        Gains à lisser, shape (n_frames, n_freq).
    attack_frames : float
        Temps d'attaque en frames STFT.
    release_frames : float
        Temps de release en frames STFT.
    freeze_threshold : float
        Seuil de puissance pour le freeze.
    power : np.ndarray, optional
        Puissance du signal, shape (n_frames, n_freq).

    Returns
    -------
    smoothed : np.ndarray
        Gains lissés, shape (n_frames, n_freq).
    """
    n_freq = gains.shape[1]

    smoother = RampSmooth(
        n_freq=n_freq,
        attack_frames=attack_frames,
        release_frames=release_frames,
        freeze_threshold=freeze_threshold,
    )

    return smoother.process_batch(gains, power)


def generate_extraction_mask(
    panning: np.ndarray,
    source_pan: float,
    width: float,
    slope: float,
    min_gain_db: float,
    attack_frames: float,
    release_frames: float,
    apply_blur: bool = True,
    apply_smoothing: bool = True,
    power: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Génère le masque d'extraction complet pour une source.

    Combine :
    1. Création de la LUT
    2. Application de la LUT au panning estimé
    3. Blur fréquentiel (optionnel)
    4. Lissage temporel (optionnel)

    Parameters
    ----------
    panning : np.ndarray
        Estimation de panning, shape (n_frames, n_freq).
        Valeurs entre -1 et 1.
    source_pan : float
        Position de la source à extraire, entre -1 et 1.
    width : float
        Largeur du masque.
    slope : float
        Pente de la transition.
    min_gain_db : float
        Gain minimum en dB.
    attack_frames : float
        Temps d'attaque en frames.
    release_frames : float
        Temps de release en frames.
    apply_blur : bool
        Si True, appliquer le blur fréquentiel.
    apply_smoothing : bool
        Si True, appliquer le lissage temporel.
    power : np.ndarray, optional
        Puissance du signal pour le freeze.

    Returns
    -------
    mask : np.ndarray
        Masque d'extraction, shape (n_frames, n_freq).
        Gains linéaires entre min_gain (linéaire) et 1.
    """
    # 1. Créer la LUT
    x_lut, lut = create_mask_lut(source_pan, width, slope, min_gain_db)

    # 2. Appliquer la LUT au panning
    mask = interpolate_lut(x_lut, lut, panning)

    # 3. Blur fréquentiel (par frame)
    if apply_blur:
        mask = apply_freq_blur(mask, exclude_dc_nyquist=True)

    # 4. Lissage temporel
    if apply_smoothing:
        mask = apply_temporal_smoothing(
            mask,
            attack_frames=attack_frames,
            release_frames=release_frames,
            power=power,
        )

    return mask
