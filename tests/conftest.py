"""
Configuration pytest partagée pour tous les tests.
Définit les fixtures communes utilisées dans les tests.
"""

import json  # Sera utilisé pour charger les fixtures JSON de test
from pathlib import Path  # Sera utilisé pour les chemins de fichiers WAV

import numpy as np
import pytest

# Chemins vers les fixtures
FIXTURES_DIR = Path(__file__).parent / "fixtures"
SIGNALS_DIR = FIXTURES_DIR / "test_signals"
PARAMS_DIR = FIXTURES_DIR / "test_params"


@pytest.fixture
def sample_rate():
    """Fréquence d'échantillonnage standard pour les tests."""
    return 48000


@pytest.fixture
def duration_seconds():
    """Durée des signaux de test en secondes."""
    return 1.0


@pytest.fixture
def n_samples(sample_rate, duration_seconds):
    """Nombre d'échantillons pour les signaux de test."""
    return int(sample_rate * duration_seconds)


@pytest.fixture
def stereo_signal(n_samples):
    """Signal stéréo de test (sinusoïde)."""
    t = np.linspace(0, 1.0, n_samples)
    freq = 440.0
    left = np.sin(2 * np.pi * freq * t)
    right = np.sin(2 * np.pi * freq * t + np.pi / 4)  # Déphasage
    return np.column_stack([left, right])


@pytest.fixture
def multichannel_signal_5_1(n_samples):
    """Signal 5.1 de test."""
    t = np.linspace(0, 1.0, n_samples)
    freq = 440.0
    signals = []
    for i in range(6):  # L, R, C, LFE, LS, RS
        phase = i * np.pi / 6
        signals.append(np.sin(2 * np.pi * freq * t + phase))
    return np.column_stack(signals)


@pytest.fixture
def white_noise_signal(n_samples):
    """Bruit blanc pour les tests."""
    return np.random.randn(n_samples).astype(np.float32)


@pytest.fixture
def sine_signal(n_samples, sample_rate):
    """Signal sinusoïdal pur."""
    freq = 1000.0
    t = np.arange(n_samples) / sample_rate
    return np.sin(2 * np.pi * freq * t).astype(np.float32)


@pytest.fixture
def default_params():
    """Paramètres par défaut pour les tests."""
    return {
        "F_xover1": 150.0,
        "F_LFE": 120.0,
        "max_sources": 11,
        "nfft": 128,
        "overlap": 0.25,
        "input_layout": "stereo",
        "output_layout": "5.1",
    }


@pytest.fixture
def upmix_params_json():
    """Paramètres upmix de base pour les tests."""
    return {
        "width": 0.18,
        "slope": 500.0,
        "min_gain": -40.0,
        "attack": 1.0,
        "pan1": 0.5,
        "gains1": [0.5, 0.3, 0.2, 0.0, 0.0, 0.0],
        "delays1": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        "release1": 50.0,
        "mute1": 0,
        "LF_gain1": 1.0,
    }


@pytest.fixture
def biquad_params():
    """Paramètres pour les tests biquad."""
    return {
        "freq": 150.0,
        "q": 0.707,
        "fs": 48000.0,
    }
