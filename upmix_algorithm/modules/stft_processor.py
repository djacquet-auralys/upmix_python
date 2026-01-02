"""
Module de traitement STFT (Short-Time Fourier Transform).

Implémente la STFT forward et inverse avec fenêtre duale sqrt(hann)
et overlap-add pour reconstruction parfaite.
"""

from typing import Optional

import numpy as np
from scipy.signal import get_window


def create_sqrt_hann_window(n: int) -> np.ndarray:
    """
    Crée une fenêtre sqrt(hann) pour analyse et synthèse.

    Cette fenêtre permet une reconstruction parfaite avec overlap-add
    quand elle est utilisée à la fois pour l'analyse et la synthèse.

    Args:
        n: Taille de la fenêtre.

    Returns:
        Fenêtre sqrt(hann) de taille n.
    """
    hann = get_window("hann", n, fftbins=False)
    return np.sqrt(hann).astype(np.float32)


def compute_latency(nfft: int, _overlap: float = 0.25) -> int:
    """
    Calcule la latence introduite par la STFT/ISTFT.

    La latence correspond à 2 fois la taille de la fenêtre (nfft)
    pour une fenêtre duale avec 25% d'overlap.

    Args:
        nfft: Taille de la FFT.
        overlap: Fraction d'overlap (défaut: 0.25).

    Returns:
        Latence en samples.
    """
    # Latence = 2 * nfft pour reconstruction parfaite avec fenêtre duale
    return 2 * nfft


class STFTProcessor:
    """
    Processeur STFT/ISTFT avec fenêtre duale sqrt(hann).

    Paramètres par défaut :
    - nfft = 128
    - overlap = 0.25 (hop_size = 32)
    - Fenêtre : sqrt(hann) pour analyse et synthèse

    Attributes:
        nfft: Taille de la FFT.
        hop_size: Pas entre les fenêtres (nfft * overlap).
        window: Fenêtre d'analyse/synthèse.
        n_freq: Nombre de bins fréquentiels (nfft/2 + 1).
    """

    def __init__(self, nfft: int = 128, overlap: float = 0.25) -> None:
        """
        Initialise le processeur STFT.

        Args:
            nfft: Taille de la FFT (défaut: 128).
            overlap: Fraction d'overlap (défaut: 0.25).
        """
        if nfft <= 0 or nfft & (nfft - 1) != 0:
            raise ValueError(
                f"nfft doit être une puissance de 2 positive, reçu: {nfft}"
            )

        if not 0 < overlap < 1:
            raise ValueError(f"overlap doit être entre 0 et 1, reçu: {overlap}")

        self.nfft = nfft
        self.overlap = overlap
        self.hop_size = int(nfft * overlap)
        self.n_freq = nfft // 2 + 1

        # Fenêtre duale sqrt(hann)
        self.window = create_sqrt_hann_window(nfft)

        # Calcul du facteur de normalisation pour overlap-add
        # Avec 25% d'overlap (hop = nfft/4), on a 4 overlaps
        self._norm_factor = self._compute_norm_factor()

    def _compute_norm_factor(self) -> float:
        """
        Calcule le facteur de normalisation pour la reconstruction.

        Returns:
            Facteur de normalisation.
        """
        # Avec fenêtre sqrt(hann) pour analyse et synthèse,
        # le facteur de normalisation est la somme des fenêtres carrées
        # divisée par le nombre d'overlaps
        n_overlaps = self.nfft // self.hop_size
        window_squared_sum: float = np.sum(self.window**2)

        # Pour une reconstruction parfaite, on divise par ce facteur
        return window_squared_sum / n_overlaps

    def forward(self, signal: np.ndarray) -> np.ndarray:
        """
        Calcule la STFT du signal.

        Args:
            signal: Signal d'entrée (1D array).

        Returns:
            STFT complexe de forme (n_frames, n_freq).
        """
        if signal.ndim != 1:
            raise ValueError("Le signal doit être un tableau 1D")

        signal = signal.astype(np.float32)
        n_samples = len(signal)

        # Calcul du nombre de frames
        n_frames = (n_samples - self.nfft) // self.hop_size + 1

        if n_frames <= 0:
            raise ValueError(
                f"Signal trop court ({n_samples} samples) pour nfft={self.nfft}"
            )

        # Initialisation de la sortie
        stft = np.zeros((n_frames, self.n_freq), dtype=np.complex64)

        # Calcul frame par frame
        for i in range(n_frames):
            start = i * self.hop_size
            end = start + self.nfft
            frame = signal[start:end] * self.window
            stft[i] = np.fft.rfft(frame)

        return stft

    def inverse(
        self, stft: np.ndarray, original_length: Optional[int] = None
    ) -> np.ndarray:
        """
        Calcule l'ISTFT (reconstruction du signal).

        Args:
            stft: STFT complexe de forme (n_frames, n_freq).
            original_length: Longueur originale du signal (optionnel).

        Returns:
            Signal reconstruit.
        """
        if stft.ndim != 2:
            raise ValueError("STFT doit être un tableau 2D (n_frames, n_freq)")

        n_frames, n_freq = stft.shape

        if n_freq != self.n_freq:
            raise ValueError(f"Nombre de bins incorrect ({n_freq} vs {self.n_freq})")

        # Calcul de la longueur du signal de sortie
        output_length = (n_frames - 1) * self.hop_size + self.nfft

        # Initialisation avec overlap-add
        signal = np.zeros(output_length, dtype=np.float32)
        norm = np.zeros(output_length, dtype=np.float32)

        # Reconstruction frame par frame
        for i in range(n_frames):
            start = i * self.hop_size
            end = start + self.nfft

            # IFFT et application de la fenêtre de synthèse
            frame = np.fft.irfft(stft[i], n=self.nfft).astype(np.float32)
            frame *= self.window

            # Overlap-add
            signal[start:end] += frame
            norm[start:end] += self.window**2

        # Normalisation pour éviter les variations d'amplitude
        # Éviter la division par zéro
        norm = np.maximum(norm, 1e-10)
        signal /= norm

        # Tronquer à la longueur originale si spécifiée
        if original_length is not None:
            signal = signal[:original_length]

        return signal

    def forward_multichannel(self, signal: np.ndarray) -> np.ndarray:
        """
        Calcule la STFT pour un signal multicanal.

        Args:
            signal: Signal d'entrée (n_samples, n_channels).

        Returns:
            STFT de forme (n_frames, n_freq, n_channels).
        """
        if signal.ndim != 2:
            raise ValueError(
                "Le signal doit être un tableau 2D (n_samples, n_channels)"
            )

        n_samples, n_channels = signal.shape

        # Calculer le nombre de frames
        n_frames = (n_samples - self.nfft) // self.hop_size + 1

        if n_frames <= 0:
            raise ValueError(
                f"Signal trop court ({n_samples} samples) pour nfft={self.nfft}"
            )

        # Initialisation
        stft = np.zeros((n_frames, self.n_freq, n_channels), dtype=np.complex64)

        # STFT par canal
        for ch in range(n_channels):
            stft[:, :, ch] = self.forward(signal[:, ch])

        return stft

    def inverse_multichannel(
        self, stft: np.ndarray, original_length: Optional[int] = None
    ) -> np.ndarray:
        """
        Calcule l'ISTFT pour un signal multicanal.

        Args:
            stft: STFT de forme (n_frames, n_freq, n_channels).
            original_length: Longueur originale du signal (optionnel).

        Returns:
            Signal reconstruit (n_samples, n_channels).
        """
        if stft.ndim != 3:
            raise ValueError(
                "STFT doit être un tableau 3D (n_frames, n_freq, n_channels)"
            )

        n_frames, _n_freq, n_channels = stft.shape

        # Calculer la longueur de sortie
        output_length = (n_frames - 1) * self.hop_size + self.nfft

        if original_length is not None:
            output_length = min(output_length, original_length)

        # Initialisation
        signal = np.zeros((output_length, n_channels), dtype=np.float32)

        # ISTFT par canal
        for ch in range(n_channels):
            signal[:, ch] = self.inverse(stft[:, :, ch], original_length)

        return signal

    def get_latency(self) -> int:
        """
        Retourne la latence en samples.

        Returns:
            Latence en samples.
        """
        return compute_latency(self.nfft, self.overlap)

    def get_frequency_bins(self, fs: float) -> np.ndarray:
        """
        Retourne les fréquences correspondant à chaque bin.

        Args:
            fs: Fréquence d'échantillonnage en Hz.

        Returns:
            Tableau des fréquences en Hz.
        """
        return np.fft.rfftfreq(self.nfft, 1 / fs)
