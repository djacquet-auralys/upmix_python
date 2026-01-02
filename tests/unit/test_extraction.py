"""
Tests unitaires pour l'extraction fréquentielle.

Module testé : upmix_algorithm.modules.extractor
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from upmix_algorithm.modules.extractor import (
    SourceExtractor,
    apply_mask_to_stft,
    extract_multiple_sources,
    extract_source,
    get_channel_angles,
    select_closest_channel,
)


class TestGetChannelAngles:
    """Tests pour get_channel_angles."""

    def test_stereo_angles(self):
        """Test : Angles stéréo normalisés."""
        angles, lfe_indices = get_channel_angles("stereo")

        # Stéréo: L=-30°, R=+30° -> normalisé par 60° -> L=-0.5, R=+0.5
        assert len(angles) == 2
        assert len(lfe_indices) == 0
        # L est à -30°, R est à +30°
        assert angles[0] < 0  # L (gauche)
        assert angles[1] > 0  # R (droite)

    def test_51_angles(self):
        """Test : Angles 5.1 avec LFE."""
        angles, lfe_indices = get_channel_angles("5.1")

        # 5.1: L, R, C, LFE, LS, RS
        assert len(angles) == 6
        assert len(lfe_indices) == 1
        assert 3 in lfe_indices  # LFE est à l'index 3

    def test_71_angles(self):
        """Test : Angles 7.1."""
        angles, lfe_indices = get_channel_angles("7.1")

        assert len(angles) == 8
        assert len(lfe_indices) == 1

    def test_angles_dtype(self):
        """Test : Type des angles."""
        angles, _ = get_channel_angles("stereo")
        assert angles.dtype == np.float32

    def test_angles_range(self):
        """Test : Angles dans la plage [-1, 1]."""
        for layout in ["stereo", "5.1", "7.1"]:
            angles, _ = get_channel_angles(layout)
            assert np.all(angles >= -1.0)
            assert np.all(angles <= 1.0)


class TestSelectClosestChannel:
    """Tests pour select_closest_channel."""

    def test_select_left_channel(self):
        """Test : Sélection canal gauche pour panning gauche."""
        # Panning tout à gauche
        panning = np.full((10, 65), -0.8, dtype=np.float32)
        channel_angles = np.array([-0.5, 0.5], dtype=np.float32)  # L, R

        selected = select_closest_channel(panning, channel_angles)

        # Doit sélectionner canal 0 (gauche)
        assert np.all(selected == 0)

    def test_select_right_channel(self):
        """Test : Sélection canal droit pour panning droit."""
        panning = np.full((10, 65), 0.8, dtype=np.float32)
        channel_angles = np.array([-0.5, 0.5], dtype=np.float32)

        selected = select_closest_channel(panning, channel_angles)

        assert np.all(selected == 1)

    def test_select_center_channel(self):
        """Test : Sélection pour panning centre."""
        panning = np.full((10, 65), 0.0, dtype=np.float32)
        # L, R, C angles
        channel_angles = np.array([-0.5, 0.5, 0.0], dtype=np.float32)

        selected = select_closest_channel(panning, channel_angles)

        # Doit sélectionner canal 2 (centre)
        assert np.all(selected == 2)

    def test_exclude_lfe(self):
        """Test : Exclusion du canal LFE."""
        panning = np.full((10, 65), 0.0, dtype=np.float32)
        # L, R, C, LFE (index 3)
        channel_angles = np.array([-0.5, 0.5, 0.0, 0.0], dtype=np.float32)

        selected = select_closest_channel(panning, channel_angles, lfe_indices=[3])

        # Ne doit jamais sélectionner l'index 3
        assert not np.any(selected == 3)
        # Doit sélectionner le centre (index 2)
        assert np.all(selected == 2)

    def test_varying_panning(self):
        """Test : Panning variable par bin."""
        n_frames, n_freq = 5, 10
        panning = np.zeros((n_frames, n_freq), dtype=np.float32)
        panning[:, :5] = -0.8  # Gauche pour premières fréquences
        panning[:, 5:] = 0.8  # Droite pour autres fréquences

        channel_angles = np.array([-0.5, 0.5], dtype=np.float32)

        selected = select_closest_channel(panning, channel_angles)

        assert np.all(selected[:, :5] == 0)  # Gauche
        assert np.all(selected[:, 5:] == 1)  # Droite

    def test_output_dtype(self):
        """Test : Type de sortie."""
        panning = np.zeros((10, 65), dtype=np.float32)
        channel_angles = np.array([-0.5, 0.5], dtype=np.float32)

        selected = select_closest_channel(panning, channel_angles)

        assert selected.dtype == np.int32


class TestApplyMaskToStft:
    """Tests pour apply_mask_to_stft."""

    def test_basic_multiplication(self):
        """Test : Multiplication basique."""
        n_frames, n_freq, n_channels = 10, 65, 2
        stft = np.ones((n_frames, n_freq, n_channels), dtype=np.complex64)
        mask = np.full((n_frames, n_freq), 0.5, dtype=np.float32)
        indices = np.zeros((n_frames, n_freq), dtype=np.int32)

        result = apply_mask_to_stft(stft, mask, indices)

        assert result.shape == (n_frames, n_freq)
        assert_allclose(np.abs(result), 0.5, rtol=1e-5)

    def test_complex_preservation(self):
        """Test : Préservation de la phase complexe."""
        n_frames, n_freq, n_channels = 5, 10, 2
        # STFT avec phase non-nulle
        stft = np.exp(
            1j * np.random.uniform(0, 2 * np.pi, (n_frames, n_freq, n_channels))
        )
        stft = stft.astype(np.complex64)
        mask = np.ones((n_frames, n_freq), dtype=np.float32)
        indices = np.zeros((n_frames, n_freq), dtype=np.int32)

        result = apply_mask_to_stft(stft, mask, indices)

        # La phase doit être préservée
        original_phase = np.angle(stft[:, :, 0])
        result_phase = np.angle(result)
        assert_allclose(result_phase, original_phase, rtol=1e-5)

    def test_channel_selection(self):
        """Test : Sélection correcte du canal."""
        n_frames, n_freq, n_channels = 5, 10, 3
        stft = np.zeros((n_frames, n_freq, n_channels), dtype=np.complex64)
        stft[:, :, 0] = 1.0  # Canal 0
        stft[:, :, 1] = 2.0  # Canal 1
        stft[:, :, 2] = 3.0  # Canal 2

        mask = np.ones((n_frames, n_freq), dtype=np.float32)

        # Sélectionner canal 1 pour toutes les frames
        indices = np.ones((n_frames, n_freq), dtype=np.int32)

        result = apply_mask_to_stft(stft, mask, indices)

        assert_allclose(np.real(result), 2.0, rtol=1e-5)

    def test_varying_indices(self):
        """Test : Indices variables par bin."""
        n_frames, n_freq, n_channels = 2, 4, 2
        stft = np.zeros((n_frames, n_freq, n_channels), dtype=np.complex64)
        stft[:, :, 0] = 1.0
        stft[:, :, 1] = 2.0

        mask = np.ones((n_frames, n_freq), dtype=np.float32)

        indices = np.array(
            [
                [0, 0, 1, 1],
                [0, 0, 1, 1],
            ],
            dtype=np.int32,
        )

        result = apply_mask_to_stft(stft, mask, indices)

        assert_allclose(np.real(result[:, :2]), 1.0, rtol=1e-5)
        assert_allclose(np.real(result[:, 2:]), 2.0, rtol=1e-5)

    def test_output_dtype(self):
        """Test : Type de sortie complex64."""
        stft = np.ones((5, 10, 2), dtype=np.complex64)
        mask = np.ones((5, 10), dtype=np.float32)
        indices = np.zeros((5, 10), dtype=np.int32)

        result = apply_mask_to_stft(stft, mask, indices)

        assert result.dtype == np.complex64


class TestExtractSource:
    """Tests pour extract_source."""

    @pytest.fixture
    def stereo_stft(self):
        """Fixture : STFT stéréo simple."""
        n_frames, n_freq = 20, 65
        stft = np.ones((n_frames, n_freq, 2), dtype=np.complex64)
        stft[:, :, 0] *= 1.0  # L
        stft[:, :, 1] *= 2.0  # R
        return stft

    @pytest.fixture
    def stereo_panning(self):
        """Fixture : Panning stéréo."""
        return np.zeros((20, 65), dtype=np.float32)  # Centre

    def test_extract_basic(self, stereo_stft, stereo_panning):
        """Test : Extraction basique."""
        result = extract_source(
            stft=stereo_stft,
            panning=stereo_panning,
            source_pan=0.0,
            width=0.5,
            slope=100.0,
            min_gain_db=-40.0,
            attack_frames=1.0,
            release_frames=50.0,
            input_layout="stereo",
            apply_blur=False,
            apply_smoothing=False,
        )

        assert result.shape == (20, 65)
        assert result.dtype == np.complex64

    def test_extract_left_source(self, stereo_stft, stereo_panning):
        """Test : Extraction source gauche."""
        # Panning indiquant signal à gauche
        panning = np.full((20, 65), -0.8, dtype=np.float32)

        result = extract_source(
            stft=stereo_stft,
            panning=panning,
            source_pan=-0.8,  # Source à gauche
            width=0.5,
            slope=100.0,
            min_gain_db=-40.0,
            attack_frames=1.0,
            release_frames=50.0,
            input_layout="stereo",
            apply_blur=False,
            apply_smoothing=False,
        )

        # Le résultat doit venir principalement du canal gauche
        # et avoir un gain proche de 1 (car source_pan = panning)
        assert np.mean(np.abs(result)) > 0.5

    def test_extract_with_smoothing(self, stereo_stft, stereo_panning):
        """Test : Extraction avec lissage temporel."""
        result = extract_source(
            stft=stereo_stft,
            panning=stereo_panning,
            source_pan=0.0,
            width=0.5,
            slope=100.0,
            min_gain_db=-40.0,
            attack_frames=1.0,
            release_frames=50.0,
            input_layout="stereo",
            apply_blur=True,
            apply_smoothing=True,
        )

        # Doit retourner un résultat valide
        assert not np.any(np.isnan(result))
        assert not np.any(np.isinf(result))


class TestExtractMultipleSources:
    """Tests pour extract_multiple_sources."""

    @pytest.fixture
    def stereo_stft(self):
        """Fixture : STFT stéréo."""
        return np.ones((20, 65, 2), dtype=np.complex64)

    @pytest.fixture
    def panning(self):
        """Fixture : Panning centre."""
        return np.zeros((20, 65), dtype=np.float32)

    def test_extract_two_sources(self, stereo_stft, panning):
        """Test : Extraction de deux sources."""
        source_params = [
            {
                "pan": -0.5,
                "width": 0.3,
                "slope": 100.0,
                "min_gain_db": -40.0,
                "attack_frames": 1.0,
                "release_frames": 50.0,
            },
            {
                "pan": 0.5,
                "width": 0.3,
                "slope": 100.0,
                "min_gain_db": -40.0,
                "attack_frames": 1.0,
                "release_frames": 50.0,
            },
        ]

        results = extract_multiple_sources(
            stft=stereo_stft,
            panning=panning,
            source_params=source_params,
            input_layout="stereo",
            apply_blur=False,
            apply_smoothing=False,
        )

        assert len(results) == 2
        for result in results:
            assert result.shape == (20, 65)

    def test_muted_source(self, stereo_stft, panning):
        """Test : Source mutée ignorée."""
        source_params = [
            {
                "pan": -0.5,
                "width": 0.3,
                "slope": 100.0,
                "min_gain_db": -40.0,
                "attack_frames": 1.0,
                "release_frames": 50.0,
                "mute": 0,  # Non mutée
            },
            {
                "pan": 0.5,
                "width": 0.3,
                "slope": 100.0,
                "min_gain_db": -40.0,
                "attack_frames": 1.0,
                "release_frames": 50.0,
                "mute": 1,  # Mutée
            },
        ]

        results = extract_multiple_sources(
            stft=stereo_stft,
            panning=panning,
            source_params=source_params,
            input_layout="stereo",
        )

        # Une seule source car l'autre est mutée
        assert len(results) == 1

    def test_all_muted(self, stereo_stft, panning):
        """Test : Toutes les sources mutées."""
        source_params = [
            {
                "pan": 0.0,
                "width": 0.3,
                "slope": 100.0,
                "min_gain_db": -40.0,
                "attack_frames": 1.0,
                "release_frames": 50.0,
                "mute": 1,
            },
        ]

        results = extract_multiple_sources(
            stft=stereo_stft,
            panning=panning,
            source_params=source_params,
            input_layout="stereo",
        )

        assert len(results) == 0

    def test_independent_extraction(self):
        """Test : Extractions indépendantes."""
        # STFT avec valeurs différentes par canal
        n_frames, n_freq = 20, 65
        stft = np.zeros((n_frames, n_freq, 2), dtype=np.complex64)
        stft[:, :, 0] = 1.0  # Canal gauche
        stft[:, :, 1] = 2.0  # Canal droit

        # Panning variable : gauche pour premières fréquences, droite pour autres
        panning = np.zeros((n_frames, n_freq), dtype=np.float32)
        panning[:, :32] = -0.8  # Gauche
        panning[:, 32:] = 0.8  # Droite

        # Deux sources : une à gauche, une à droite
        source_params = [
            {
                "pan": -0.8,  # Source gauche
                "width": 0.5,
                "slope": 100.0,
                "min_gain_db": -40.0,
                "attack_frames": 1.0,
                "release_frames": 50.0,
            },
            {
                "pan": 0.8,  # Source droite
                "width": 0.5,
                "slope": 100.0,
                "min_gain_db": -40.0,
                "attack_frames": 1.0,
                "release_frames": 50.0,
            },
        ]

        results = extract_multiple_sources(
            stft=stft,
            panning=panning,
            source_params=source_params,
            input_layout="stereo",
            apply_blur=False,
            apply_smoothing=False,
        )

        # Les résultats doivent être différents car masques différents
        # Source gauche aura plus de gain sur les fréquences gauches
        # Source droite aura plus de gain sur les fréquences droites
        assert not np.allclose(results[0], results[1])


class TestSourceExtractor:
    """Tests pour la classe SourceExtractor."""

    def test_init(self):
        """Test : Initialisation."""
        extractor = SourceExtractor("stereo")

        assert extractor.input_layout == "stereo"
        assert len(extractor.channel_angles) == 2
        assert len(extractor.lfe_indices) == 0

    def test_init_51(self):
        """Test : Initialisation avec 5.1."""
        extractor = SourceExtractor("5.1")

        assert len(extractor.channel_angles) == 6
        assert len(extractor.lfe_indices) == 1

    def test_extract_method(self):
        """Test : Méthode extract."""
        extractor = SourceExtractor("stereo")
        stft = np.ones((10, 65, 2), dtype=np.complex64)
        panning = np.zeros((10, 65), dtype=np.float32)

        result = extractor.extract(
            stft=stft,
            panning=panning,
            source_pan=0.0,
            width=0.5,
            slope=100.0,
            min_gain_db=-40.0,
            attack_frames=1.0,
            release_frames=50.0,
        )

        assert result.shape == (10, 65)

    def test_extract_batch_method(self):
        """Test : Méthode extract_batch."""
        extractor = SourceExtractor("stereo")
        stft = np.ones((10, 65, 2), dtype=np.complex64)
        panning = np.zeros((10, 65), dtype=np.float32)

        source_params = [
            {
                "pan": -0.5,
                "width": 0.3,
                "slope": 100.0,
                "min_gain_db": -40.0,
                "attack_frames": 1.0,
                "release_frames": 50.0,
            },
        ]

        results = extractor.extract_batch(
            stft=stft,
            panning=panning,
            source_params=source_params,
        )

        assert len(results) == 1

    def test_get_channel_info(self):
        """Test : Méthode get_channel_info."""
        extractor = SourceExtractor("5.1")

        info = extractor.get_channel_info()

        assert info["layout"] == "5.1"
        assert info["n_channels"] == 6
        assert info["n_valid_channels"] == 5  # Sans LFE


class TestEdgeCases:
    """Tests pour cas limites."""

    def test_silent_signal(self):
        """Test : Signal silencieux."""
        stft = np.zeros((10, 65, 2), dtype=np.complex64)
        panning = np.zeros((10, 65), dtype=np.float32)

        result = extract_source(
            stft=stft,
            panning=panning,
            source_pan=0.0,
            width=0.5,
            slope=100.0,
            min_gain_db=-40.0,
            attack_frames=1.0,
            release_frames=50.0,
            input_layout="stereo",
        )

        # Signal silencieux -> résultat silencieux
        assert_allclose(result, 0.0, atol=1e-10)

    def test_all_channels_same(self):
        """Test : Tous les canaux identiques."""
        stft = np.ones((10, 65, 2), dtype=np.complex64) * (1.0 + 0.5j)
        panning = np.zeros((10, 65), dtype=np.float32)

        result = extract_source(
            stft=stft,
            panning=panning,
            source_pan=0.0,
            width=1.0,
            slope=100.0,
            min_gain_db=-40.0,
            attack_frames=1.0,
            release_frames=50.0,
            input_layout="stereo",
            apply_blur=False,
            apply_smoothing=False,
        )

        # Doit fonctionner sans erreur
        assert not np.any(np.isnan(result))

    def test_extreme_panning(self):
        """Test : Panning aux extrêmes."""
        stft = np.ones((10, 65, 2), dtype=np.complex64)

        # Panning à -1 (extrême gauche)
        panning = np.full((10, 65), -1.0, dtype=np.float32)

        result = extract_source(
            stft=stft,
            panning=panning,
            source_pan=-1.0,
            width=0.2,
            slope=100.0,
            min_gain_db=-40.0,
            attack_frames=1.0,
            release_frames=50.0,
            input_layout="stereo",
        )

        assert not np.any(np.isnan(result))

    def test_multicanal_extraction(self):
        """Test : Extraction depuis 5.1."""
        n_frames, n_freq = 10, 65
        stft = np.ones((n_frames, n_freq, 6), dtype=np.complex64)
        panning = np.zeros((n_frames, n_freq), dtype=np.float32)

        result = extract_source(
            stft=stft,
            panning=panning,
            source_pan=0.0,
            width=0.5,
            slope=100.0,
            min_gain_db=-40.0,
            attack_frames=1.0,
            release_frames=50.0,
            input_layout="5.1",
        )

        assert result.shape == (n_frames, n_freq)
