"""
Filtres IIR Biquad selon les formules du W3C Audio EQ Cookbook.

Ce module implémente les filtres biquad standards :
- LPF (Low Pass Filter)
- HPF (High Pass Filter)
- PK (Peaking EQ)
- LOW_SHELF (Low Shelf)
- HIGH_SHELF (High Shelf)

Référence : https://www.w3.org/TR/audio-eq-cookbook/#formulae
"""

from enum import Enum
from typing import Tuple

import numpy as np
from scipy.signal import freqz, lfilter, lfilter_zi


class FilterType(Enum):
    """Types de filtres supportés."""

    LPF = "lowpass"
    HPF = "highpass"
    PK = "peaking"
    LOW_SHELF = "lowshelf"
    HIGH_SHELF = "highshelf"


def compute_biquad_coeffs(
    freq: float,
    q: float,
    fs: float,
    filter_type: FilterType,
    gain_db: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calcule les coefficients biquad selon le W3C Audio EQ Cookbook.

    Les coefficients sont normalisés (a0 = 1).

    Args:
        freq: Fréquence centrale/coupure en Hz.
        q: Facteur de qualité (typiquement 0.707 pour Butterworth).
        fs: Fréquence d'échantillonnage en Hz.
        filter_type: Type de filtre (LPF, HPF, PK, LOW_SHELF, HIGH_SHELF).
        gain_db: Gain en dB (utilisé uniquement pour PK, LOW_SHELF, HIGH_SHELF).

    Returns:
        Tuple (b, a) avec:
            - b: Coefficients du numérateur [b0, b1, b2]
            - a: Coefficients du dénominateur [1, a1, a2] (normalisé)

    Raises:
        ValueError: Si la fréquence ou Q est invalide.
    """
    if freq <= 0 or freq >= fs / 2:
        raise ValueError(
            f"La fréquence ({freq} Hz) doit être entre 0 et Nyquist ({fs/2} Hz)"
        )

    if q <= 0:
        raise ValueError(f"Le facteur Q ({q}) doit être positif")

    # Calcul des paramètres intermédiaires
    omega0 = 2 * np.pi * freq / fs
    cos_omega0 = np.cos(omega0)
    sin_omega0 = np.sin(omega0)
    alpha = sin_omega0 / (2 * q)

    # Pour les filtres avec gain (PK, shelves)
    A = 10 ** (gain_db / 40)  # A = 10^(dBgain/40) pour peaking et shelving

    # Calcul des coefficients selon le type de filtre
    if filter_type == FilterType.LPF:
        # Low Pass Filter
        b0 = (1 - cos_omega0) / 2
        b1 = 1 - cos_omega0
        b2 = (1 - cos_omega0) / 2
        a0 = 1 + alpha
        a1 = -2 * cos_omega0
        a2 = 1 - alpha

    elif filter_type == FilterType.HPF:
        # High Pass Filter
        b0 = (1 + cos_omega0) / 2
        b1 = -(1 + cos_omega0)
        b2 = (1 + cos_omega0) / 2
        a0 = 1 + alpha
        a1 = -2 * cos_omega0
        a2 = 1 - alpha

    elif filter_type == FilterType.PK:
        # Peaking EQ
        b0 = 1 + alpha * A
        b1 = -2 * cos_omega0
        b2 = 1 - alpha * A
        a0 = 1 + alpha / A
        a1 = -2 * cos_omega0
        a2 = 1 - alpha / A

    elif filter_type == FilterType.LOW_SHELF:
        # Low Shelf
        sqrt_A = np.sqrt(A)
        b0 = A * ((A + 1) - (A - 1) * cos_omega0 + 2 * sqrt_A * alpha)
        b1 = 2 * A * ((A - 1) - (A + 1) * cos_omega0)
        b2 = A * ((A + 1) - (A - 1) * cos_omega0 - 2 * sqrt_A * alpha)
        a0 = (A + 1) + (A - 1) * cos_omega0 + 2 * sqrt_A * alpha
        a1 = -2 * ((A - 1) + (A + 1) * cos_omega0)
        a2 = (A + 1) + (A - 1) * cos_omega0 - 2 * sqrt_A * alpha

    elif filter_type == FilterType.HIGH_SHELF:
        # High Shelf
        sqrt_A = np.sqrt(A)
        b0 = A * ((A + 1) + (A - 1) * cos_omega0 + 2 * sqrt_A * alpha)
        b1 = -2 * A * ((A - 1) + (A + 1) * cos_omega0)
        b2 = A * ((A + 1) + (A - 1) * cos_omega0 - 2 * sqrt_A * alpha)
        a0 = (A + 1) - (A - 1) * cos_omega0 + 2 * sqrt_A * alpha
        a1 = 2 * ((A - 1) - (A + 1) * cos_omega0)
        a2 = (A + 1) - (A - 1) * cos_omega0 - 2 * sqrt_A * alpha

    else:
        raise ValueError(f"Type de filtre non supporté: {filter_type}")

    # Normalisation par a0
    b = np.array([b0 / a0, b1 / a0, b2 / a0], dtype=np.float32)
    a = np.array([1.0, a1 / a0, a2 / a0], dtype=np.float32)

    return b, a


class BiquadFilter:
    """
    Filtre IIR Biquad avec état persistant.

    Cette classe encapsule un filtre biquad et maintient son état
    entre les appels successifs à process().

    Attributes:
        b: Coefficients du numérateur.
        a: Coefficients du dénominateur.
        freq: Fréquence centrale/coupure.
        q: Facteur de qualité.
        fs: Fréquence d'échantillonnage.
        filter_type: Type de filtre.
        gain_db: Gain en dB (pour PK, LOW_SHELF, HIGH_SHELF).
    """

    def __init__(
        self,
        freq: float,
        q: float,
        fs: float,
        filter_type: FilterType,
        gain_db: float = 0.0,
    ) -> None:
        """
        Initialise le filtre biquad.

        Args:
            freq: Fréquence centrale/coupure en Hz.
            q: Facteur de qualité.
            fs: Fréquence d'échantillonnage en Hz.
            filter_type: Type de filtre.
            gain_db: Gain en dB (pour PK, LOW_SHELF, HIGH_SHELF).
        """
        self.freq = freq
        self.q = q
        self.fs = fs
        self.filter_type = filter_type
        self.gain_db = gain_db

        # Calcul des coefficients
        self.b, self.a = compute_biquad_coeffs(freq, q, fs, filter_type, gain_db)

        # État initial du filtre (pour continuité entre appels)
        self._zi = lfilter_zi(self.b, self.a).astype(np.float32)
        self._state = self._zi * 0.0  # État nul au départ

    def process(self, signal: np.ndarray) -> np.ndarray:
        """
        Applique le filtre au signal.

        L'état du filtre est maintenu entre les appels successifs.

        Args:
            signal: Signal d'entrée (1D array).

        Returns:
            Signal filtré (même longueur que l'entrée).
        """
        if signal.ndim != 1:
            raise ValueError("Le signal doit être un tableau 1D")

        # Conversion en float32 si nécessaire
        signal = signal.astype(np.float32)

        # Application du filtre avec état
        output, self._state = lfilter(self.b, self.a, signal, zi=self._state)

        return output.astype(np.float32)

    def reset(self) -> None:
        """Réinitialise l'état du filtre à zéro."""
        self._state = self._zi * 0.0

    def get_frequency_response(
        self, n_points: int = 512
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calcule la réponse en fréquence du filtre.

        Args:
            n_points: Nombre de points de la réponse.

        Returns:
            Tuple (frequencies, magnitude_db) avec:
                - frequencies: Fréquences en Hz
                - magnitude_db: Magnitude en dB
        """
        w, h = freqz(self.b, self.a, worN=n_points)
        frequencies = w * self.fs / (2 * np.pi)
        magnitude_db = 20 * np.log10(np.abs(h) + 1e-10)

        return frequencies, magnitude_db


class CascadeBiquadFilter:
    """
    Cascade de filtres biquad pour obtenir un ordre supérieur.

    Pour obtenir -6dB à la fréquence de coupure avec un ordre 4,
    on utilise 2 biquads en cascade.
    """

    def __init__(
        self,
        freq: float,
        q: float,
        fs: float,
        filter_type: FilterType,
        n_stages: int = 2,
        gain_db: float = 0.0,
    ) -> None:
        """
        Initialise la cascade de filtres.

        Args:
            freq: Fréquence centrale/coupure en Hz.
            q: Facteur de qualité.
            fs: Fréquence d'échantillonnage en Hz.
            filter_type: Type de filtre.
            n_stages: Nombre de biquads en cascade (défaut: 2 pour ordre 4).
            gain_db: Gain en dB (pour PK, LOW_SHELF, HIGH_SHELF).
        """
        self.freq = freq
        self.q = q
        self.fs = fs
        self.filter_type = filter_type
        self.n_stages = n_stages
        self.gain_db = gain_db

        # Création des filtres en cascade
        self._filters = [
            BiquadFilter(freq, q, fs, filter_type, gain_db) for _ in range(n_stages)
        ]

    def process(self, signal: np.ndarray) -> np.ndarray:
        """
        Applique la cascade de filtres au signal.

        Args:
            signal: Signal d'entrée (1D array).

        Returns:
            Signal filtré.
        """
        output = signal
        for filt in self._filters:
            output = filt.process(output)
        return output

    def reset(self) -> None:
        """Réinitialise l'état de tous les filtres."""
        for filt in self._filters:
            filt.reset()

    def get_frequency_response(
        self, n_points: int = 512
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calcule la réponse en fréquence de la cascade.

        Args:
            n_points: Nombre de points.

        Returns:
            Tuple (frequencies, magnitude_db).
        """
        # Réponse du premier filtre
        frequencies, magnitude_db = self._filters[0].get_frequency_response(n_points)

        # Ajouter la contribution des autres filtres
        for filt in self._filters[1:]:
            _, mag_db = filt.get_frequency_response(n_points)
            magnitude_db += mag_db

        return frequencies, magnitude_db
