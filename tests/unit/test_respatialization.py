"""
Tests unitaires pour la respatialisation.

Module testé : upmix_algorithm.modules.respatializer
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from upmix_algorithm.modules.respatializer import (
    Respatializer,
    add_lf_mono1_to_sources,
    add_lfe_to_output,
    apply_delay,
    apply_gain_and_delay,
    compute_default_gains,
    get_available_output_layouts,
    get_output_layout_info,
    ms_to_samples,
    parse_source_params,
    spatialize_sources,
)


@pytest.fixture
def sample_rate():
    """Fixture : Fréquence d'échantillonnage."""
    return 48000.0


class TestGetOutputLayoutInfo:
    """Tests pour get_output_layout_info."""

    def test_stereo_info(self):
        """Test : Info layout stéréo."""
        info = get_output_layout_info("stereo")

        assert info["n_channels"] == 2
        assert info["n_speakers"] == 2
        assert info["n_lfe"] == 0
        assert len(info["lfe_indices"]) == 0

    def test_51_info(self):
        """Test : Info layout 5.1."""
        info = get_output_layout_info("5.1")

        assert info["n_channels"] == 6
        assert info["n_speakers"] == 5  # Sans LFE
        assert info["n_lfe"] == 1
        assert 3 in info["lfe_indices"]  # LFE à l'index 3

    def test_71_info(self):
        """Test : Info layout 7.1."""
        info = get_output_layout_info("7.1")

        assert info["n_channels"] == 8
        assert info["n_speakers"] == 7
        assert info["n_lfe"] == 1

    def test_labels(self):
        """Test : Labels des canaux."""
        info = get_output_layout_info("5.1")

        assert "L" in info["labels"]
        assert "R" in info["labels"]
        assert "C" in info["labels"]
        assert "LFE" in info["labels"]


class TestMsToSamples:
    """Tests pour ms_to_samples."""

    def test_basic_conversion(self, sample_rate):
        """Test : Conversion basique."""
        # 1 ms à 48000 Hz = 48 samples
        assert ms_to_samples(1.0, sample_rate) == 48

    def test_zero_delay(self, sample_rate):
        """Test : Délai nul."""
        assert ms_to_samples(0.0, sample_rate) == 0

    def test_fractional_rounding(self, sample_rate):
        """Test : Arrondi des valeurs fractionnaires."""
        # 0.5 ms à 48000 Hz = 24 samples
        assert ms_to_samples(0.5, sample_rate) == 24

        # 0.52 ms à 48000 Hz ≈ 24.96 -> arrondi à 25
        assert ms_to_samples(0.52, sample_rate) == 25

    def test_large_delay(self, sample_rate):
        """Test : Grand délai."""
        # 100 ms à 48000 Hz = 4800 samples
        assert ms_to_samples(100.0, sample_rate) == 4800


class TestApplyDelay:
    """Tests pour apply_delay."""

    def test_no_delay(self):
        """Test : Pas de délai."""
        signal = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
        result = apply_delay(signal, 0)

        assert_allclose(result, signal)

    def test_basic_delay(self):
        """Test : Délai basique."""
        signal = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
        result = apply_delay(signal, 2)

        expected = np.array([0.0, 0.0, 1.0, 2.0, 3.0], dtype=np.float32)
        assert_allclose(result, expected)

    def test_delay_preserves_shape(self):
        """Test : Le délai préserve la forme."""
        signal = np.ones(100, dtype=np.float32)
        result = apply_delay(signal, 10)

        assert result.shape == signal.shape

    def test_delay_longer_than_signal(self):
        """Test : Délai plus long que le signal."""
        signal = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        result = apply_delay(signal, 10)

        # Signal entièrement à zéro
        assert_allclose(result, np.zeros(3, dtype=np.float32))

    def test_negative_delay(self):
        """Test : Délai négatif (traité comme zéro)."""
        signal = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        result = apply_delay(signal, -5)

        # Pas de modification
        assert_allclose(result, signal)


class TestApplyGainAndDelay:
    """Tests pour apply_gain_and_delay."""

    def test_basic_gain_delay(self, sample_rate):
        """Test : Application basique gain + délai."""
        source = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
        gains = np.array([0.5, 1.0], dtype=np.float32)
        delays_ms = np.array([0.0, 0.0], dtype=np.float32)

        result = apply_gain_and_delay(source, gains, delays_ms, sample_rate)

        assert result.shape == (5, 2)
        assert_allclose(result[:, 0], source * 0.5)
        assert_allclose(result[:, 1], source * 1.0)

    def test_zero_gain(self, sample_rate):
        """Test : Gain à zéro."""
        source = np.ones(10, dtype=np.float32)
        gains = np.array([0.0, 1.0], dtype=np.float32)
        delays_ms = np.array([0.0, 0.0], dtype=np.float32)

        result = apply_gain_and_delay(source, gains, delays_ms, sample_rate)

        assert_allclose(result[:, 0], 0.0)
        assert_allclose(result[:, 1], 1.0)


class TestSpatializeSources:
    """Tests pour spatialize_sources."""

    def test_single_source(self, sample_rate):
        """Test : Une seule source."""
        sources = [np.ones(100, dtype=np.float32)]
        gains = [np.array([0.5, 0.5, 0.0], dtype=np.float32)]
        delays = [np.array([0.0, 0.0, 0.0], dtype=np.float32)]

        result = spatialize_sources(sources, gains, delays, sample_rate, 3)

        assert result.shape == (100, 3)
        assert_allclose(result[:, 0], 0.5)
        assert_allclose(result[:, 1], 0.5)
        assert_allclose(result[:, 2], 0.0)

    def test_two_sources(self, sample_rate):
        """Test : Deux sources sommées."""
        sources = [
            np.ones(100, dtype=np.float32),
            np.ones(100, dtype=np.float32) * 2,
        ]
        gains = [
            np.array([1.0, 0.0], dtype=np.float32),
            np.array([0.0, 1.0], dtype=np.float32),
        ]
        delays = [
            np.array([0.0, 0.0], dtype=np.float32),
            np.array([0.0, 0.0], dtype=np.float32),
        ]

        result = spatialize_sources(sources, gains, delays, sample_rate, 2)

        assert_allclose(result[:, 0], 1.0)  # Source 1 uniquement
        assert_allclose(result[:, 1], 2.0)  # Source 2 uniquement

    def test_no_sources_error(self, sample_rate):
        """Test : Erreur si pas de sources."""
        with pytest.raises(ValueError, match="Aucune source"):
            spatialize_sources([], [], [], sample_rate, 2)


class TestAddLfeToOutput:
    """Tests pour add_lfe_to_output."""

    def test_basic_lfe_routing(self):
        """Test : Routage basique du LFE."""
        output = np.zeros((100, 6), dtype=np.float32)
        lfe_signal = np.ones(100, dtype=np.float32) * 0.5

        result = add_lfe_to_output(output, lfe_signal, 3)

        assert_allclose(result[:, 3], 0.5)
        assert_allclose(result[:, 0], 0.0)  # Autres canaux inchangés

    def test_lfe_overwrites(self):
        """Test : LFE remplace le contenu existant."""
        output = np.ones((100, 6), dtype=np.float32)
        lfe_signal = np.zeros(100, dtype=np.float32)

        result = add_lfe_to_output(output, lfe_signal, 3)

        assert_allclose(result[:, 3], 0.0)  # LFE à zéro
        assert_allclose(result[:, 0], 1.0)  # Autres canaux préservés


class TestAddLfMono1ToSources:
    """Tests pour add_lf_mono1_to_sources."""

    def test_basic_addition(self):
        """Test : Addition basique."""
        sources = [np.ones(100, dtype=np.float32)]
        lf_mono1 = np.ones(100, dtype=np.float32) * 0.5
        lf_gains = [1.0]

        result = add_lf_mono1_to_sources(sources, lf_mono1, lf_gains, 0)

        # 1.0 + 0.5 * 1.0 = 1.5
        assert_allclose(result[0], 1.5)

    def test_with_latency(self):
        """Test : Avec latence."""
        sources = [np.zeros(100, dtype=np.float32)]
        lf_mono1 = np.ones(100, dtype=np.float32)
        lf_gains = [1.0]

        result = add_lf_mono1_to_sources(sources, lf_mono1, lf_gains, 10)

        # Les 10 premiers échantillons doivent rester à 0
        assert_allclose(result[0][:10], 0.0)
        # Le reste doit avoir le LF ajouté
        assert_allclose(result[0][10:], 1.0)

    def test_different_gains(self):
        """Test : Gains différents par source."""
        sources = [
            np.zeros(100, dtype=np.float32),
            np.zeros(100, dtype=np.float32),
        ]
        lf_mono1 = np.ones(100, dtype=np.float32)
        lf_gains = [0.5, 2.0]

        result = add_lf_mono1_to_sources(sources, lf_mono1, lf_gains, 0)

        assert_allclose(result[0], 0.5)
        assert_allclose(result[1], 2.0)


class TestParseSourceParams:
    """Tests pour parse_source_params."""

    def test_basic_parsing(self):
        """Test : Parsing basique."""
        upmix_params = {
            "gains1": [0.5, 0.5, 0.0, 0.0, 0.0],
            "delays1": [0.0, 0.0, 0.0, 0.0, 0.0],
            "LF_gain1": 0.8,
            "mute1": 0,
        }

        gains, delays, lf_gains, mutes = parse_source_params(upmix_params, 1, 5)

        assert len(gains) == 1
        assert len(delays) == 1
        assert_allclose(gains[0], [0.5, 0.5, 0.0, 0.0, 0.0])
        assert lf_gains[0] == 0.8
        assert mutes[0] is False

    def test_muted_source(self):
        """Test : Source mutée."""
        upmix_params = {
            "mute1": 1,
        }

        _, _, _, mutes = parse_source_params(upmix_params, 1, 5)

        assert mutes[0] is True

    def test_default_values(self):
        """Test : Valeurs par défaut."""
        upmix_params = {}  # Pas de paramètres

        gains, delays, lf_gains, mutes = parse_source_params(upmix_params, 1, 5)

        # Gains par défaut = 1.0
        assert_allclose(gains[0], np.ones(5))
        # Délais par défaut = 0.0
        assert_allclose(delays[0], np.zeros(5))
        # LF_gain par défaut = 1.0
        assert lf_gains[0] == 1.0
        # Non muté par défaut
        assert mutes[0] is False

    def test_multiple_sources(self):
        """Test : Plusieurs sources."""
        upmix_params = {
            "gains1": [1.0, 0.0],
            "gains2": [0.0, 1.0],
            "mute1": 0,
            "mute2": 1,
        }

        gains, _, _, mutes = parse_source_params(upmix_params, 2, 2)

        assert len(gains) == 2
        assert mutes[0] is False
        assert mutes[1] is True


class TestRespatializer:
    """Tests pour la classe Respatializer."""

    def test_init(self, sample_rate):
        """Test : Initialisation."""
        resp = Respatializer("5.1", sample_rate)

        assert resp.output_layout == "5.1"
        assert resp.sample_rate == sample_rate
        assert resp.n_output_channels == 6
        assert resp.n_speakers == 5

    def test_spatialize_basic(self, sample_rate):
        """Test : Spatialisation basique."""
        resp = Respatializer("stereo", sample_rate)

        sources = [np.ones(100, dtype=np.float32)]
        gains = [np.array([0.7, 0.7], dtype=np.float32)]
        delays = [np.array([0.0, 0.0], dtype=np.float32)]

        result = resp.spatialize(sources, gains, delays)

        assert result.shape == (100, 2)
        assert_allclose(result[:, 0], 0.7)
        assert_allclose(result[:, 1], 0.7)

    def test_spatialize_with_lfe(self, sample_rate):
        """Test : Spatialisation avec LFE."""
        resp = Respatializer("5.1", sample_rate)

        sources = [np.ones(100, dtype=np.float32)]
        gains = [np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)]
        delays = [np.zeros(6, dtype=np.float32)]
        lfe = np.ones(100, dtype=np.float32) * 0.5

        result = resp.spatialize(sources, gains, delays, lfe_signal=lfe)

        assert result.shape == (100, 6)
        assert_allclose(result[:, 3], 0.5)  # LFE

    def test_spatialize_no_sources_with_lfe(self, sample_rate):
        """Test : Pas de sources mais LFE fourni."""
        resp = Respatializer("5.1", sample_rate)

        lfe = np.ones(100, dtype=np.float32)
        result = resp.spatialize([], [], [], lfe_signal=lfe)

        assert result.shape == (100, 6)
        assert_allclose(result[:, 3], 1.0)  # LFE

    def test_spatialize_from_params(self, sample_rate):
        """Test : Spatialisation depuis paramètres JSON."""
        resp = Respatializer("stereo", sample_rate)

        sources = [np.ones(100, dtype=np.float32)]
        upmix_params = {
            "gains1": [0.5, 0.5],
            "delays1": [0.0, 0.0],
            "LF_gain1": 1.0,
        }

        result = resp.spatialize_from_params(sources, upmix_params)

        assert result.shape == (100, 2)


class TestComputeDefaultGains:
    """Tests pour compute_default_gains."""

    def test_center_source(self):
        """Test : Source au centre."""
        gains = compute_default_gains(0.0, "stereo")

        # Source au centre -> gains égaux L et R
        assert len(gains) == 2
        assert gains[0] == gains[1]

    def test_left_source(self):
        """Test : Source à gauche."""
        gains = compute_default_gains(-1.0, "stereo")

        # Source à gauche -> plus de gain sur L
        assert gains[0] > gains[1]

    def test_right_source(self):
        """Test : Source à droite."""
        gains = compute_default_gains(1.0, "stereo")

        # Source à droite -> plus de gain sur R
        assert gains[1] > gains[0]

    def test_51_layout(self):
        """Test : Layout 5.1."""
        gains = compute_default_gains(0.0, "5.1")

        # 5 HP (sans LFE)
        assert len(gains) == 5


class TestGetAvailableOutputLayouts:
    """Tests pour get_available_output_layouts."""

    def test_returns_list(self):
        """Test : Retourne une liste."""
        layouts = get_available_output_layouts()

        assert isinstance(layouts, list)
        assert len(layouts) > 0

    def test_common_layouts(self):
        """Test : Layouts courants présents."""
        layouts = get_available_output_layouts()

        assert "stereo" in layouts
        assert "5.1" in layouts
        assert "7.1" in layouts


class TestEdgeCases:
    """Tests pour cas limites."""

    def test_very_short_signal(self, sample_rate):
        """Test : Signal très court."""
        resp = Respatializer("stereo", sample_rate)

        sources = [np.array([1.0], dtype=np.float32)]
        gains = [np.array([1.0, 1.0], dtype=np.float32)]
        delays = [np.array([0.0, 0.0], dtype=np.float32)]

        result = resp.spatialize(sources, gains, delays)

        assert result.shape == (1, 2)

    def test_large_gains(self, sample_rate):
        """Test : Gains élevés (vérification pas de saturation explicite)."""
        resp = Respatializer("stereo", sample_rate)

        sources = [np.ones(100, dtype=np.float32)]
        gains = [np.array([5.0, 5.0], dtype=np.float32)]
        delays = [np.array([0.0, 0.0], dtype=np.float32)]

        result = resp.spatialize(sources, gains, delays)

        # Pas de clipping automatique, valeurs > 1
        assert np.max(result) > 1.0

    def test_all_muted_sources(self, sample_rate):
        """Test : Toutes sources mutées avec LFE."""
        resp = Respatializer("5.1", sample_rate)

        sources = [np.ones(100, dtype=np.float32)]
        upmix_params = {"mute1": 1}
        lfe = np.ones(100, dtype=np.float32)

        result = resp.spatialize_from_params(sources, upmix_params, lfe_signal=lfe)

        # Seul le LFE doit être présent
        assert_allclose(result[:, 3], 1.0)
        assert_allclose(result[:, 0], 0.0)
