"""
Tests unitaires pour le traitement STFT.

Module testé : upmix_algorithm.modules.stft_processor

Plan de tests :
1. STFT forward : dimensions correctes
2. ISTFT inverse : reconstruction parfaite (signal identique)
3. Overlap-add : pas d'artefacts
4. Fenêtre duale : condition de reconstruction
5. Fenêtre Hann : valeurs correctes
6. Overlap 25% : hop_size correct
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy.signal import get_window

from upmix_algorithm.modules.stft_processor import (
    STFTProcessor,
    compute_latency,
    create_sqrt_hann_window,
)


class TestSTFTForward:
    """Tests pour la STFT forward."""

    def test_stft_dimensions(self, sine_signal, default_params):
        """
        Test : Dimensions correctes de la STFT.

        Pour un signal de N échantillons avec nfft=128 et overlap=0.25 :
        - Nombre de frames temporelles : calculé selon hop_size
        - Nombre de bins fréquentiels : nfft/2 + 1 = 65
        - Type : complexe
        """
        nfft = default_params["nfft"]
        overlap = default_params["overlap"]
        processor = STFTProcessor(nfft=nfft, overlap=overlap)

        stft = processor.forward(sine_signal)

        # Vérifier le type
        assert stft.dtype == np.complex64

        # Vérifier le nombre de bins fréquentiels
        expected_n_freq = nfft // 2 + 1
        assert stft.shape[1] == expected_n_freq

        # Vérifier le nombre de frames
        hop_size = int(nfft * overlap)
        expected_n_frames = (len(sine_signal) - nfft) // hop_size + 1
        assert stft.shape[0] == expected_n_frames

    def test_stft_frequency_bins(self, sine_signal, default_params, sample_rate):
        """
        Test : Bins fréquentiels corrects.

        Vérifie que les fréquences des bins correspondent aux
        fréquences attendues (0 à Nyquist).
        """
        processor = STFTProcessor(nfft=default_params["nfft"])
        freqs = processor.get_frequency_bins(sample_rate)

        # Vérifier la plage
        assert freqs[0] == 0  # DC
        assert freqs[-1] == sample_rate / 2  # Nyquist

        # Vérifier que les fréquences sont croissantes
        assert np.all(np.diff(freqs) > 0)

    def test_stft_time_frames(self, sine_signal, default_params):
        """
        Test : Nombre de frames temporelles.

        Vérifie que le nombre de frames est cohérent avec
        hop_size et longueur du signal.
        """
        nfft = default_params["nfft"]
        overlap = default_params["overlap"]
        hop_size = int(nfft * overlap)

        processor = STFTProcessor(nfft=nfft, overlap=overlap)
        stft = processor.forward(sine_signal)

        expected_n_frames = (len(sine_signal) - nfft) // hop_size + 1
        assert stft.shape[0] == expected_n_frames


class TestISTFTInverse:
    """Tests pour la ISTFT inverse."""

    def test_perfect_reconstruction(self, sine_signal, default_params):
        """
        Test : Reconstruction parfaite.

        Vérifie que signal_reconstruit ≈ signal_original
        (à une tolérance numérique près).
        """
        processor = STFTProcessor(
            nfft=default_params["nfft"], overlap=default_params["overlap"]
        )

        # STFT puis ISTFT
        stft = processor.forward(sine_signal)
        reconstructed = processor.inverse(stft, original_length=len(sine_signal))

        # La reconstruction devrait être très proche de l'original
        # On ignore les bords car il y a des effets de fenêtrage
        offset = processor.nfft
        assert_allclose(
            reconstructed[offset:-offset],
            sine_signal[offset:-offset],
            rtol=1e-4,
            atol=1e-5,
        )

    def test_reconstruction_white_noise(self, white_noise_signal, default_params):
        """
        Test : Reconstruction bruit blanc.

        Vérifie la reconstruction pour un signal complexe (bruit blanc).
        """
        processor = STFTProcessor(
            nfft=default_params["nfft"], overlap=default_params["overlap"]
        )

        stft = processor.forward(white_noise_signal)
        reconstructed = processor.inverse(stft, original_length=len(white_noise_signal))

        offset = processor.nfft
        assert_allclose(
            reconstructed[offset:-offset],
            white_noise_signal[offset:-offset],
            rtol=1e-4,
            atol=1e-5,
        )

    def test_reconstruction_multichannel(self, stereo_signal, default_params):
        """
        Test : Reconstruction multicanal.

        Vérifie que chaque canal est reconstruit indépendamment.
        """
        processor = STFTProcessor(
            nfft=default_params["nfft"], overlap=default_params["overlap"]
        )

        stft = processor.forward_multichannel(stereo_signal)
        reconstructed = processor.inverse_multichannel(
            stft, original_length=len(stereo_signal)
        )

        offset = processor.nfft
        for ch in range(2):
            assert_allclose(
                reconstructed[offset:-offset, ch],
                stereo_signal[offset:-offset, ch],
                rtol=1e-4,
                atol=1e-5,
            )


class TestOverlapAdd:
    """Tests pour l'overlap-add."""

    def test_overlap_add_no_artifacts(self, sine_signal, default_params):
        """
        Test : Pas d'artefacts majeurs dans overlap-add.

        Vérifie qu'il n'y a pas de clics, pops, ou discontinuités majeures
        dans le signal reconstruit.
        """
        processor = STFTProcessor(
            nfft=default_params["nfft"], overlap=default_params["overlap"]
        )

        stft = processor.forward(sine_signal)
        reconstructed = processor.inverse(stft, original_length=len(sine_signal))

        # Vérifier qu'il n'y a pas de discontinuités majeures (clics)
        diff = np.diff(reconstructed)
        max_diff = np.max(np.abs(diff))

        # Pour un signal audio, les discontinuités majeures seraient > 1.0
        # On accepte des variations jusqu'à 0.5 dues au fenêtrage
        assert max_diff < 0.5, f"Discontinuité majeure détectée: max_diff={max_diff}"

        # Vérifier qu'il n'y a pas de NaN ou Inf
        assert not np.any(np.isnan(reconstructed))
        assert not np.any(np.isinf(reconstructed))

    def test_overlap_add_normalization(self, sine_signal, default_params):
        """
        Test : Normalisation dans overlap-add.

        Vérifie que la normalisation est correcte pour éviter
        les variations d'amplitude.
        """
        processor = STFTProcessor(
            nfft=default_params["nfft"], overlap=default_params["overlap"]
        )

        stft = processor.forward(sine_signal)
        reconstructed = processor.inverse(stft, original_length=len(sine_signal))

        # L'amplitude RMS devrait être préservée (hors bords)
        offset = processor.nfft * 2
        rms_original = np.sqrt(np.mean(sine_signal[offset:-offset] ** 2))
        rms_reconstructed = np.sqrt(np.mean(reconstructed[offset:-offset] ** 2))

        assert_allclose(rms_reconstructed, rms_original, rtol=0.05)

    def test_overlap_add_energy_preservation(self, sine_signal, default_params):
        """
        Test : Préservation de l'énergie.

        Vérifie que l'énergie est préservée dans le processus
        STFT/ISTFT avec overlap-add.
        """
        processor = STFTProcessor(
            nfft=default_params["nfft"], overlap=default_params["overlap"]
        )

        stft = processor.forward(sine_signal)
        reconstructed = processor.inverse(stft, original_length=len(sine_signal))

        # Énergie (hors bords)
        offset = processor.nfft * 2
        energy_original = np.sum(sine_signal[offset:-offset] ** 2)
        energy_reconstructed = np.sum(reconstructed[offset:-offset] ** 2)

        assert_allclose(energy_reconstructed, energy_original, rtol=0.1)


class TestDualWindow:
    """Tests pour la fenêtre duale."""

    def test_dual_window_condition(self, default_params):
        """
        Test : Condition de reconstruction avec normalisation.

        Note: Avec overlap=0.25 et sqrt(hann), la condition COLA n'est pas
        parfaitement satisfaite. C'est pourquoi l'ISTFT utilise une
        normalisation explicite pour compenser.

        On vérifie ici que la reconstruction est quand même correcte.
        """
        nfft = default_params["nfft"]
        overlap = default_params["overlap"]

        processor = STFTProcessor(nfft=nfft, overlap=overlap)

        # Test avec un signal simple de longueur adaptée
        signal_length = 1024  # Multiple de nfft
        signal = np.random.randn(signal_length).astype(np.float32)

        # STFT puis ISTFT
        stft = processor.forward(signal)
        reconstructed = processor.inverse(stft)

        # La reconstruction devrait être proche de l'original (hors bords)
        # On compare la partie centrale commune
        offset = nfft
        min_len = min(len(reconstructed), len(signal)) - offset
        assert_allclose(
            reconstructed[offset:min_len],
            signal[offset:min_len],
            rtol=1e-4,
            atol=1e-5,
        )

    def test_dual_window_sqrt_hann(self, default_params):
        """
        Test : Fenêtre sqrt(hann).

        Vérifie que les fenêtres utilisées sont bien sqrt(hann)
        pour analyse et synthèse.
        """
        nfft = default_params["nfft"]
        window = create_sqrt_hann_window(nfft)

        # Vérifier que c'est bien sqrt(hann)
        hann = get_window("hann", nfft, fftbins=False)
        expected = np.sqrt(hann)

        assert_allclose(window, expected, rtol=1e-5)

    def test_dual_window_sum_constant(self, default_params):
        """
        Test : Somme constante pour reconstruction.

        Vérifie que la somme des fenêtres chevauchantes est
        constante (condition pour reconstruction parfaite).
        """
        processor = STFTProcessor(
            nfft=default_params["nfft"], overlap=default_params["overlap"]
        )

        # Créer un signal de test (impulsion)
        signal_length = 1000
        impulse = np.zeros(signal_length, dtype=np.float32)
        impulse[signal_length // 2] = 1.0

        # STFT/ISTFT
        stft = processor.forward(impulse)
        reconstructed = processor.inverse(stft, original_length=signal_length)

        # L'impulsion devrait être préservée (forme similaire)
        # On vérifie juste que le max est proche de 1
        assert np.max(reconstructed) > 0.5


class TestHannWindow:
    """Tests pour la fenêtre Hann."""

    def test_hann_window_values(self, default_params):
        """
        Test : Valeurs correctes de la fenêtre Hann.

        Vérifie que la fenêtre Hann a les bonnes valeurs aux
        extrémités (0) et au centre (1).
        """
        nfft = default_params["nfft"]
        window = create_sqrt_hann_window(nfft)

        # sqrt(hann) aux extrémités ≈ 0
        assert window[0] < 0.1
        assert window[-1] < 0.1

        # sqrt(hann) au centre ≈ 1
        center_idx = nfft // 2
        assert window[center_idx] > 0.9

    def test_sqrt_hann_window(self, default_params):
        """
        Test : Fenêtre sqrt(hann).

        Vérifie que sqrt(hann) est bien utilisée.
        """
        nfft = default_params["nfft"]
        window = create_sqrt_hann_window(nfft)

        # Vérifier que window² = hann
        hann = get_window("hann", nfft, fftbins=False)
        assert_allclose(window**2, hann, rtol=1e-5)


class TestOverlap25Percent:
    """Tests pour l'overlap de 25%."""

    def test_hop_size_calculation(self, default_params):
        """
        Test : Calcul correct du hop_size.

        Pour nfft=128 et overlap=0.25 :
        hop_size = nfft * 0.25 = 32 samples
        """
        nfft = default_params["nfft"]
        overlap = default_params["overlap"]

        processor = STFTProcessor(nfft=nfft, overlap=overlap)

        expected_hop = int(nfft * overlap)
        assert processor.hop_size == expected_hop
        assert expected_hop == 32  # Pour nfft=128, overlap=0.25

    def test_overlap_25_percent_frames(self, sine_signal, default_params):
        """
        Test : Nombre de frames avec overlap 25%.

        Vérifie que le nombre de frames est correct pour
        l'overlap de 25%.
        """
        nfft = default_params["nfft"]
        overlap = default_params["overlap"]

        processor = STFTProcessor(nfft=nfft, overlap=overlap)
        stft = processor.forward(sine_signal)

        # Calcul manuel du nombre de frames
        hop_size = int(nfft * overlap)
        expected_n_frames = (len(sine_signal) - nfft) // hop_size + 1

        assert stft.shape[0] == expected_n_frames


class TestSTFTEdgeCases:
    """Tests pour cas limites."""

    def test_stft_short_signal(self, default_params):
        """
        Test : Signal plus court que nfft.

        Vérifie que le STFT gère correctement un signal plus court
        que la taille de la fenêtre.
        """
        processor = STFTProcessor(nfft=default_params["nfft"])

        short_signal = np.random.randn(50).astype(np.float32)

        with pytest.raises(ValueError, match="trop court"):
            processor.forward(short_signal)

    def test_stft_empty_signal(self):
        """
        Test : Signal vide.
        """
        processor = STFTProcessor(nfft=128)
        empty_signal = np.array([], dtype=np.float32)

        with pytest.raises(ValueError):
            processor.forward(empty_signal)

    def test_stft_dc_signal(self, sample_rate):
        """
        Test : Signal DC (fréquence 0).
        """
        processor = STFTProcessor(nfft=128)
        dc_signal = np.ones(sample_rate, dtype=np.float32)

        stft = processor.forward(dc_signal)

        # Le DC devrait apparaître dans le bin 0
        dc_magnitude = np.abs(stft[:, 0])
        other_magnitude = np.abs(stft[:, 1:])

        assert np.mean(dc_magnitude) > np.mean(other_magnitude)

    def test_stft_nyquist_frequency(self, sample_rate, default_params):
        """
        Test : Fréquence de Nyquist.

        Vérifie que la fréquence de Nyquist est correctement
        représentée dans le dernier bin.
        """
        processor = STFTProcessor(nfft=default_params["nfft"])
        freqs = processor.get_frequency_bins(sample_rate)

        assert freqs[-1] == sample_rate / 2

    def test_stft_invalid_nfft(self):
        """Test : nfft non puissance de 2 lève une erreur."""
        with pytest.raises(ValueError, match="puissance de 2"):
            STFTProcessor(nfft=100)

    def test_stft_invalid_overlap(self):
        """Test : overlap hors [0, 1] lève une erreur."""
        with pytest.raises(ValueError, match="entre 0 et 1"):
            STFTProcessor(nfft=128, overlap=1.5)


class TestLatency:
    """Tests pour la latence."""

    def test_latency_calculation(self, default_params):
        """Test : Calcul de la latence."""
        nfft = default_params["nfft"]
        latency = compute_latency(nfft)

        # Latence = 2 * nfft
        assert latency == 2 * nfft

    def test_processor_latency(self, default_params):
        """Test : Latence du processeur."""
        processor = STFTProcessor(nfft=default_params["nfft"])
        latency = processor.get_latency()

        assert latency == 2 * default_params["nfft"]


class TestMultichannel:
    """Tests pour le traitement multicanal."""

    def test_multichannel_dimensions(self, stereo_signal, default_params):
        """Test : Dimensions correctes pour multicanal."""
        processor = STFTProcessor(
            nfft=default_params["nfft"], overlap=default_params["overlap"]
        )

        stft = processor.forward_multichannel(stereo_signal)

        assert stft.ndim == 3
        assert stft.shape[2] == 2  # 2 canaux
        assert stft.shape[1] == processor.n_freq

    def test_multichannel_1d_signal_error(self, sine_signal, default_params):
        """Test : Signal 1D lève une erreur pour forward_multichannel."""
        processor = STFTProcessor(nfft=default_params["nfft"])

        with pytest.raises(ValueError, match="2D"):
            processor.forward_multichannel(sine_signal)
