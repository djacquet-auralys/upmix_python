# -*- coding: utf-8 -*-
"""
Tests unitaires pour re_model_light.py

Tests pour le module léger de calcul de vecteur d'énergie
optimisé pour l'estimation de panning.
"""

import os
import sys

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_almost_equal

# Ajouter les chemins nécessaires
_test_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.dirname(os.path.dirname(_test_dir))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from upmix_algorithm.modules.re_model_light import (
    compute_energy_vector,
    energy_vector_to_angle,
    estimate_panning,
    estimate_source_width,
    get_available_layouts,
    get_energy_vector_magnitude,
    get_layout_info,
    get_speaker_unit_vectors,
)


class TestGetSpeakerUnitVectors:
    """Tests pour get_speaker_unit_vectors."""

    def test_stereo_layout(self):
        """Test vecteurs unitaires pour stéréo."""
        unit_vectors, labels, lfe_indices = get_speaker_unit_vectors("stereo")

        assert labels == ["L", "R"]
        assert lfe_indices == []
        assert unit_vectors.shape == (2, 2)

        # L = -30°, R = +30°
        # x = sin(az), y = cos(az)
        expected_L = np.array([np.sin(np.deg2rad(-30)), np.cos(np.deg2rad(-30))])
        expected_R = np.array([np.sin(np.deg2rad(30)), np.cos(np.deg2rad(30))])

        assert_array_almost_equal(unit_vectors[0], expected_L, decimal=5)
        assert_array_almost_equal(unit_vectors[1], expected_R, decimal=5)

    def test_51_layout(self):
        """Test vecteurs unitaires pour 5.1."""
        unit_vectors, labels, lfe_indices = get_speaker_unit_vectors("5.1")

        assert len(labels) == 6
        assert "LFE" in labels
        assert lfe_indices == [3]  # LFE est en position 3 dans 5.1
        assert unit_vectors.shape == (6, 2)

    def test_unit_vectors_are_normalized(self):
        """Vérifie que les vecteurs sont bien unitaires."""
        unit_vectors, _, _ = get_speaker_unit_vectors("7.1")
        norms = np.linalg.norm(unit_vectors, axis=1)
        assert_array_almost_equal(norms, np.ones(len(norms)), decimal=10)

    def test_unknown_layout_raises(self):
        """Vérifie qu'un layout inconnu lève une erreur."""
        with pytest.raises(ValueError):
            get_speaker_unit_vectors("unknown_layout")


class TestComputeEnergyVector:
    """Tests pour compute_energy_vector."""

    @pytest.fixture
    def stereo_vectors(self):
        """Vecteurs unitaires stéréo."""
        unit_vectors, _, _ = get_speaker_unit_vectors("stereo")
        return unit_vectors

    def test_equal_gains_gives_center(self, stereo_vectors):
        """Gains égaux -> vecteur centré (vers l'avant)."""
        gains = np.array([1.0, 1.0])
        re = compute_energy_vector(gains, stereo_vectors)

        # Avec gains égaux L et R, le vecteur pointe vers l'avant (y positif, x ~0)
        assert re.shape == (2,)
        assert abs(re[0]) < 0.01  # x proche de 0
        assert re[1] > 0.8  # y positif fort

    def test_left_only_gives_left(self, stereo_vectors):
        """Gain uniquement à gauche -> vecteur vers la gauche."""
        gains = np.array([1.0, 0.0])
        re = compute_energy_vector(gains, stereo_vectors)

        # L = -30°, donc x < 0
        assert re[0] < 0  # x négatif = gauche

    def test_right_only_gives_right(self, stereo_vectors):
        """Gain uniquement à droite -> vecteur vers la droite."""
        gains = np.array([0.0, 1.0])
        re = compute_energy_vector(gains, stereo_vectors)

        # R = +30°, donc x > 0
        assert re[0] > 0  # x positif = droite

    def test_2d_input(self, stereo_vectors):
        """Test avec entrée 2D (n_points, n_channels)."""
        gains = np.array(
            [
                [1.0, 0.0],  # tout à gauche
                [1.0, 1.0],  # centre
                [0.0, 1.0],  # tout à droite
            ]
        )
        re = compute_energy_vector(gains, stereo_vectors)

        assert re.shape == (3, 2)
        assert re[0, 0] < 0  # gauche
        assert abs(re[1, 0]) < 0.01  # centre
        assert re[2, 0] > 0  # droite

    def test_3d_input(self, stereo_vectors):
        """Test avec entrée 3D (n_frames, n_freq, n_channels)."""
        n_frames, n_freq = 10, 65
        gains = np.random.rand(n_frames, n_freq, 2).astype(np.float32)

        re = compute_energy_vector(gains, stereo_vectors)

        assert re.shape == (n_frames, n_freq, 2)
        assert re.dtype == np.float32

    def test_zero_gains_handled(self, stereo_vectors):
        """Test que les gains nuls ne causent pas de division par zéro."""
        gains = np.array([0.0, 0.0])
        re = compute_energy_vector(gains, stereo_vectors, epsilon=1e-12)

        # Avec epsilon, on obtient un résultat sans NaN
        assert not np.any(np.isnan(re))

    def test_lfe_exclusion(self):
        """Test que les canaux LFE sont exclus."""
        # 5.1: L, R, C, LFE, LS, RS
        unit_vectors, labels, lfe_indices = get_speaker_unit_vectors("5.1")

        # Gains: LFE très fort, autres faibles
        gains = np.array([0.1, 0.1, 0.1, 100.0, 0.1, 0.1])

        # Sans exclusion LFE
        re_with_lfe = compute_energy_vector(gains, unit_vectors, lfe_indices=None)

        # Avec exclusion LFE
        re_without_lfe = compute_energy_vector(
            gains, unit_vectors, lfe_indices=lfe_indices
        )

        # Les résultats doivent être différents
        assert not np.allclose(re_with_lfe, re_without_lfe)


class TestEnergyVectorToAngle:
    """Tests pour energy_vector_to_angle."""

    def test_center_vector(self):
        """Vecteur vers l'avant -> angle 0."""
        re = np.array([0.0, 1.0])  # +y = avant
        pan = energy_vector_to_angle(re, normalize_range=360.0)
        assert abs(pan) < 0.01

    def test_left_vector_stereo(self):
        """Vecteur vers -30° en stéréo -> pan = -1."""
        angle_rad = np.deg2rad(-30)
        re = np.array([np.sin(angle_rad), np.cos(angle_rad)])
        pan = energy_vector_to_angle(re, normalize_range=60.0)
        assert_allclose(pan, -1.0, atol=0.01)

    def test_right_vector_stereo(self):
        """Vecteur vers +30° en stéréo -> pan = +1."""
        angle_rad = np.deg2rad(30)
        re = np.array([np.sin(angle_rad), np.cos(angle_rad)])
        pan = energy_vector_to_angle(re, normalize_range=60.0)
        assert_allclose(pan, 1.0, atol=0.01)

    def test_clipping(self):
        """Vérifie que les valeurs sont clippées à [-1, 1]."""
        # Angle de 45° avec normalisation 60° devrait être clippé
        angle_rad = np.deg2rad(45)
        re = np.array([np.sin(angle_rad), np.cos(angle_rad)])
        pan = energy_vector_to_angle(re, normalize_range=60.0)
        assert pan <= 1.0
        assert pan >= -1.0

    def test_batch_processing(self):
        """Test avec plusieurs vecteurs."""
        re = np.array(
            [
                [np.sin(np.deg2rad(-30)), np.cos(np.deg2rad(-30))],
                [0.0, 1.0],
                [np.sin(np.deg2rad(30)), np.cos(np.deg2rad(30))],
            ]
        )
        pan = energy_vector_to_angle(re, normalize_range=60.0)

        assert pan.shape == (3,)
        assert_allclose(pan[0], -1.0, atol=0.01)
        assert abs(pan[1]) < 0.01
        assert_allclose(pan[2], 1.0, atol=0.01)


class TestEstimatePanning:
    """Tests pour estimate_panning (fonction principale)."""

    def test_stereo_left_signal(self):
        """Signal uniquement à gauche en stéréo."""
        # STFT avec signal uniquement sur L
        n_frames, n_freq = 10, 65
        stft_mag = np.zeros((n_frames, n_freq, 2), dtype=np.float32)
        stft_mag[:, :, 0] = 1.0  # L = 1, R = 0

        pan = estimate_panning(stft_mag, "stereo")

        assert pan.shape == (n_frames, n_freq)
        # Tout devrait être vers la gauche (-1)
        assert np.all(pan < -0.9)

    def test_stereo_right_signal(self):
        """Signal uniquement à droite en stéréo."""
        n_frames, n_freq = 10, 65
        stft_mag = np.zeros((n_frames, n_freq, 2), dtype=np.float32)
        stft_mag[:, :, 1] = 1.0  # L = 0, R = 1

        pan = estimate_panning(stft_mag, "stereo")

        assert pan.shape == (n_frames, n_freq)
        # Tout devrait être vers la droite (+1)
        assert np.all(pan > 0.9)

    def test_stereo_center_signal(self):
        """Signal centré en stéréo."""
        n_frames, n_freq = 10, 65
        stft_mag = np.ones((n_frames, n_freq, 2), dtype=np.float32)

        pan = estimate_panning(stft_mag, "stereo")

        # Devrait être proche de 0 (centre)
        assert np.all(np.abs(pan) < 0.1)

    def test_51_excludes_lfe(self):
        """Vérifie que le LFE est exclu en 5.1."""
        n_frames, n_freq = 10, 65
        # 5.1: L, R, C, LFE, LS, RS
        stft_mag = np.zeros((n_frames, n_freq, 6), dtype=np.float32)
        stft_mag[:, :, 3] = 100.0  # LFE très fort
        stft_mag[:, :, 2] = 1.0  # Centre léger

        pan = estimate_panning(stft_mag, "5.1")

        # Le panning devrait être proche du centre (C), pas affecté par le LFE
        assert pan.shape == (n_frames, n_freq)
        assert np.all(np.abs(pan) < 0.1)

    def test_output_range(self):
        """Vérifie que la sortie est toujours dans [-1, 1]."""
        n_frames, n_freq = 100, 65
        stft_mag = np.random.rand(n_frames, n_freq, 2).astype(np.float32) * 10

        pan = estimate_panning(stft_mag, "stereo")

        assert np.all(pan >= -1.0)
        assert np.all(pan <= 1.0)

    def test_dtype_float32(self):
        """Vérifie que la sortie est en float32."""
        n_frames, n_freq = 10, 65
        stft_mag = np.ones((n_frames, n_freq, 2), dtype=np.float64)

        pan = estimate_panning(stft_mag, "stereo")

        assert pan.dtype == np.float32


class TestEnergyVectorMagnitude:
    """Tests pour get_energy_vector_magnitude."""

    def test_unit_vector(self):
        """Vecteur unitaire -> magnitude = 1."""
        re = np.array([0.0, 1.0])
        mag = get_energy_vector_magnitude(re)
        assert_allclose(mag, 1.0, atol=1e-6)

    def test_zero_vector(self):
        """Vecteur nul -> magnitude = 0."""
        re = np.array([0.0, 0.0])
        mag = get_energy_vector_magnitude(re)
        assert_allclose(mag, 0.0, atol=1e-6)

    def test_batch_vectors(self):
        """Test avec plusieurs vecteurs."""
        re = np.array(
            [
                [0.0, 1.0],
                [0.5, 0.5],
                [0.0, 0.0],
            ]
        )
        mag = get_energy_vector_magnitude(re)

        assert mag.shape == (3,)
        assert_allclose(mag[0], 1.0, atol=1e-6)
        assert_allclose(mag[1], np.sqrt(0.5), atol=1e-6)
        assert_allclose(mag[2], 0.0, atol=1e-6)


class TestEstimateSourceWidth:
    """Tests pour estimate_source_width."""

    def test_point_source(self):
        """Source ponctuelle -> faible largeur."""
        n_frames, n_freq = 10, 65
        # Signal uniquement sur un HP
        stft_mag = np.zeros((n_frames, n_freq, 2), dtype=np.float32)
        stft_mag[:, :, 0] = 1.0

        width = estimate_source_width(stft_mag, "stereo")

        assert width.shape == (n_frames, n_freq)
        # Source ponctuelle = faible largeur
        assert np.all(width < 30)  # moins de 30°

    def test_diffuse_source(self):
        """Source diffuse -> grande largeur."""
        n_frames, n_freq = 10, 65
        # Signal égal sur tous les HP
        stft_mag = np.ones((n_frames, n_freq, 2), dtype=np.float32)

        width = estimate_source_width(stft_mag, "stereo")

        # Pour stéréo avec gains égaux, on attend ~30° de largeur
        # (car les HP sont à ±30°)
        assert np.all(width > 10)


class TestLayoutUtilities:
    """Tests pour les fonctions utilitaires de layout."""

    def test_get_available_layouts(self):
        """Vérifie que les layouts principaux sont disponibles."""
        layouts = get_available_layouts()

        assert "stereo" in layouts
        assert "5.1" in layouts
        assert "7.1" in layouts

    def test_get_layout_info_stereo(self):
        """Info du layout stéréo."""
        info = get_layout_info("stereo")

        assert info["n_channels"] == 2
        assert info["n_lfe"] == 0
        assert info["n_fullrange"] == 2
        assert info["is_stereo"] is True
        assert info["labels"] == ["L", "R"]

    def test_get_layout_info_51(self):
        """Info du layout 5.1."""
        info = get_layout_info("5.1")

        assert info["n_channels"] == 6
        assert info["n_lfe"] == 1
        assert info["n_fullrange"] == 5
        assert info["is_stereo"] is False
        assert 3 in info["lfe_indices"]


class TestIntegrationWithSTFT:
    """Tests d'intégration avec des données STFT réalistes."""

    def test_realistic_stereo_signal(self):
        """Test avec un signal stéréo simulé réaliste."""
        # Simuler un signal qui panne de gauche à droite
        n_frames = 100
        n_freq = 65

        # Panning linéaire de -1 (gauche) à +1 (droite) sur les frames
        pan_target = np.linspace(-1, 1, n_frames)

        # Convertir en gains L/R (loi de puissance constante simplifiée)
        gain_L = np.sqrt((1 - pan_target) / 2)
        gain_R = np.sqrt((1 + pan_target) / 2)

        # Créer STFT magnitudes
        stft_mag = np.zeros((n_frames, n_freq, 2), dtype=np.float32)
        stft_mag[:, :, 0] = gain_L[:, np.newaxis]
        stft_mag[:, :, 1] = gain_R[:, np.newaxis]

        # Estimer le panning
        pan_estimated = estimate_panning(stft_mag, "stereo")

        # Moyenner sur les fréquences
        pan_mean = np.mean(pan_estimated, axis=1)

        # Vérifier que le panning estimé suit la tendance
        # (pas parfait à cause de la non-linéarité, mais proche)
        correlation = np.corrcoef(pan_target, pan_mean)[0, 1]
        assert correlation > 0.95, f"Corrélation trop faible: {correlation}"

    def test_multichannel_71(self):
        """Test avec un signal 7.1."""
        n_frames, n_freq = 50, 65
        # 7.1: L, R, C, LFE, LS, RS, LB, RB
        n_channels = 8

        stft_mag = np.random.rand(n_frames, n_freq, n_channels).astype(np.float32)

        pan = estimate_panning(stft_mag, "7.1")

        assert pan.shape == (n_frames, n_freq)
        assert pan.dtype == np.float32
        assert np.all(pan >= -1.0)
        assert np.all(pan <= 1.0)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
