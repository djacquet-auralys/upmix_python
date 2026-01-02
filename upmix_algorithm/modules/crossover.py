"""
Module de crossover audio.

Implémente les filtres de crossover pour séparer les signaux en bandes
de fréquences (basses et hautes) et la somme à puissance constante.
"""

from typing import List, Optional, Tuple

import numpy as np

from .biquad_filter import CascadeBiquadFilter, FilterType


def apply_crossover(
    signal: np.ndarray,
    freq: float,
    fs: float,
    q: float = 0.707,
    n_stages: int = 2,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Applique un crossover à un signal mono.

    Sépare le signal en deux bandes :
    - Basses fréquences (< freq)
    - Hautes fréquences (> freq)

    Les filtres utilisent 2 biquads en cascade (ordre 4) pour obtenir
    -6dB à la fréquence de coupure.

    Args:
        signal: Signal d'entrée (1D array).
        freq: Fréquence de coupure en Hz (F_xover1).
        fs: Fréquence d'échantillonnage en Hz.
        q: Facteur de qualité des filtres (défaut: 0.707).
        n_stages: Nombre de biquads en cascade (défaut: 2 pour ordre 4).

    Returns:
        Tuple (low_freq, high_freq) avec les deux bandes de fréquences.
    """
    if signal.ndim != 1:
        raise ValueError("Le signal doit être un tableau 1D")

    # Conversion en float32
    signal = signal.astype(np.float32)

    # Création des filtres passe-bas et passe-haut
    lpf = CascadeBiquadFilter(freq, q, fs, FilterType.LPF, n_stages=n_stages)
    hpf = CascadeBiquadFilter(freq, q, fs, FilterType.HPF, n_stages=n_stages)

    # Application des filtres
    low_freq = lpf.process(signal)
    high_freq = hpf.process(signal)

    return low_freq, high_freq


def apply_crossover_multichannel(
    signal: np.ndarray,
    freq: float,
    fs: float,
    lfe_channel_indices: Optional[List[int]] = None,
    q: float = 0.707,
    n_stages: int = 2,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Applique un crossover à un signal multicanal.

    Les canaux LFE sont exclus du traitement.

    Args:
        signal: Signal d'entrée (n_samples, n_channels).
        freq: Fréquence de coupure en Hz.
        fs: Fréquence d'échantillonnage en Hz.
        lfe_channel_indices: Indices des canaux LFE à exclure.
        q: Facteur de qualité des filtres.
        n_stages: Nombre de biquads en cascade.

    Returns:
        Tuple (low_freq, high_freq) avec les bandes par canal.
        Les canaux LFE sont mis à zéro.
    """
    if signal.ndim != 2:
        raise ValueError("Le signal doit être un tableau 2D (n_samples, n_channels)")

    if lfe_channel_indices is None:
        lfe_channel_indices = []

    n_samples, n_channels = signal.shape
    low_freq = np.zeros_like(signal)
    high_freq = np.zeros_like(signal)

    for ch in range(n_channels):
        if ch in lfe_channel_indices:
            # LFE exclu du crossover, on garde le signal original
            # (sera traité séparément)
            continue

        ch_low, ch_high = apply_crossover(signal[:, ch], freq, fs, q, n_stages)
        low_freq[:, ch] = ch_low
        high_freq[:, ch] = ch_high

    return low_freq, high_freq


def sum_power_constant(signals: List[np.ndarray]) -> np.ndarray:
    """
    Somme des signaux à puissance constante.

    Formule : result = sqrt(sum(signal_i²))

    Cette méthode préserve l'énergie totale lors de la somme de signaux.

    Args:
        signals: Liste de signaux à sommer (tous de même longueur).

    Returns:
        Signal résultant de la somme à puissance constante.
    """
    if not signals:
        raise ValueError("La liste de signaux ne peut pas être vide")

    # Vérifier que tous les signaux ont la même longueur
    length = len(signals[0])
    for i, sig in enumerate(signals):
        if len(sig) != length:
            raise ValueError(
                f"Signal {i} a une longueur différente ({len(sig)} vs {length})"
            )

    # Conversion en float32
    signals = [sig.astype(np.float32) for sig in signals]

    # Calcul de la somme des carrés
    sum_squared = np.zeros(length, dtype=np.float32)
    for sig in signals:
        sum_squared += sig**2

    # Racine carrée pour la somme à puissance constante
    result = np.sqrt(sum_squared)

    return result


def compute_lf_mono1_stereo(
    left_lowfreq: np.ndarray, right_lowfreq: np.ndarray
) -> np.ndarray:
    """
    Calcule LF_mono1 pour stéréo.

    Pour stéréo : LF_mono1 = (L_lowfreq + R_lowfreq) * 0.707

    Args:
        left_lowfreq: Signal basses fréquences canal gauche.
        right_lowfreq: Signal basses fréquences canal droit.

    Returns:
        Signal LF_mono1.
    """
    return (left_lowfreq + right_lowfreq) * 0.707


def compute_lf_mono1_multichannel(
    low_freq_signals: List[np.ndarray],
) -> np.ndarray:
    """
    Calcule LF_mono1 pour multicanal.

    Formule : LF_mono1 = sqrt(sum(all_lowfreq²))

    Args:
        low_freq_signals: Liste des signaux basses fréquences (hors LFE).

    Returns:
        Signal LF_mono1.
    """
    return sum_power_constant(low_freq_signals)


class Crossover:
    """
    Classe de crossover avec état pour traitement continu.

    Cette classe encapsule les filtres de crossover et peut être utilisée
    pour traiter des signaux de manière continue (streaming).
    """

    def __init__(
        self,
        freq: float,
        fs: float,
        n_channels: int = 2,
        lfe_channel_indices: Optional[List[int]] = None,
        q: float = 0.707,
        n_stages: int = 2,
    ) -> None:
        """
        Initialise le crossover.

        Args:
            freq: Fréquence de coupure en Hz.
            fs: Fréquence d'échantillonnage en Hz.
            n_channels: Nombre de canaux.
            lfe_channel_indices: Indices des canaux LFE à exclure.
            q: Facteur de qualité.
            n_stages: Nombre de biquads en cascade.
        """
        self.freq = freq
        self.fs = fs
        self.n_channels = n_channels
        self.lfe_channel_indices = lfe_channel_indices or []
        self.q = q
        self.n_stages = n_stages

        # Création des filtres pour chaque canal (hors LFE)
        self._lpf_filters = {}
        self._hpf_filters = {}

        for ch in range(n_channels):
            if ch not in self.lfe_channel_indices:
                self._lpf_filters[ch] = CascadeBiquadFilter(
                    freq, q, fs, FilterType.LPF, n_stages
                )
                self._hpf_filters[ch] = CascadeBiquadFilter(
                    freq, q, fs, FilterType.HPF, n_stages
                )

    def process(self, signal: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Applique le crossover au signal.

        Args:
            signal: Signal d'entrée (n_samples,) ou (n_samples, n_channels).

        Returns:
            Tuple (low_freq, high_freq, lf_mono1).
        """
        # Gestion signal mono
        if signal.ndim == 1:
            signal = signal.reshape(-1, 1)

        n_samples, n_channels = signal.shape

        if n_channels != self.n_channels:
            raise ValueError(
                f"Nombre de canaux incorrect ({n_channels} vs {self.n_channels})"
            )

        low_freq = np.zeros_like(signal, dtype=np.float32)
        high_freq = np.zeros_like(signal, dtype=np.float32)

        # Traitement par canal
        for ch in range(n_channels):
            if ch in self.lfe_channel_indices:
                continue

            low_freq[:, ch] = self._lpf_filters[ch].process(signal[:, ch])
            high_freq[:, ch] = self._hpf_filters[ch].process(signal[:, ch])

        # Calcul de LF_mono1
        active_channels = [
            ch for ch in range(n_channels) if ch not in self.lfe_channel_indices
        ]

        if len(active_channels) == 2:
            # Stéréo
            lf_mono1 = compute_lf_mono1_stereo(
                low_freq[:, active_channels[0]], low_freq[:, active_channels[1]]
            )
        else:
            # Multicanal
            low_freq_list = [low_freq[:, ch] for ch in active_channels]
            lf_mono1 = compute_lf_mono1_multichannel(low_freq_list)

        return low_freq, high_freq, lf_mono1

    def reset(self) -> None:
        """Réinitialise l'état de tous les filtres."""
        for filt in self._lpf_filters.values():
            filt.reset()
        for filt in self._hpf_filters.values():
            filt.reset()
