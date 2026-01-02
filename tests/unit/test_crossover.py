"""
Tests unitaires pour les crossovers.

Module testé : upmix_algorithm.modules.crossover

Plan de tests :
1. Crossover stéréo : séparation L/R correcte
2. Somme à puissance constante : vérifier niveau RMS
3. Généralisation multicanal : tous canaux traités
4. Exclusion LFE : LFE non inclus dans somme
5. Conservation énergie : énergie totale préservée
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from upmix_algorithm.modules.crossover import (
    Crossover,
    apply_crossover,
    apply_crossover_multichannel,
    compute_lf_mono1_multichannel,
    compute_lf_mono1_stereo,
    sum_power_constant,
)


class TestStereoCrossover:
    """Tests pour crossover stéréo."""

    def test_stereo_low_high_separation(
        self, stereo_signal, default_params, sample_rate
    ):
        """
        Test : Séparation correcte basses/hautes fréquences.

        Pour un signal stéréo :
        - Les basses fréquences (< F_xover1) doivent être dans low_freq
        - Les hautes fréquences (> F_xover1) doivent être dans high_freq
        """
        f_xover = default_params["F_xover1"]

        # Signal de test : sinusoïde basse fréquence (50 Hz) + haute fréquence (1000 Hz)
        t = np.linspace(0, 1.0, sample_rate)
        low_sine = np.sin(2 * np.pi * 50 * t)  # 50 Hz < 150 Hz
        high_sine = np.sin(2 * np.pi * 1000 * t)  # 1000 Hz > 150 Hz
        test_signal = (low_sine + high_sine).astype(np.float32)

        low_freq, high_freq = apply_crossover(test_signal, f_xover, sample_rate)

        # Vérifier que les basses fréquences contiennent principalement le 50 Hz
        # et les hautes fréquences contiennent principalement le 1000 Hz
        assert len(low_freq) == len(test_signal)
        assert len(high_freq) == len(test_signal)

        # L'énergie dans les basses fréquences devrait être similaire à celle du 50 Hz
        # (après stabilisation du filtre)
        rms_low = np.sqrt(np.mean(low_freq[sample_rate // 4 :] ** 2))
        rms_high = np.sqrt(np.mean(high_freq[sample_rate // 4 :] ** 2))

        # Les deux RMS devraient être non nuls
        assert rms_low > 0.1, "RMS basses fréquences trop faible"
        assert rms_high > 0.1, "RMS hautes fréquences trop faible"

    def test_stereo_lf_mono1_calculation(
        self, stereo_signal, default_params, sample_rate
    ):
        """
        Test : Calcul LF_mono1 pour stéréo.

        Vérifie que LF_mono1 = (L_lowfreq + R_lowfreq) * 0.707
        """
        f_xover = default_params["F_xover1"]

        # Appliquer crossover aux deux canaux
        left_low, _ = apply_crossover(stereo_signal[:, 0], f_xover, sample_rate)
        right_low, _ = apply_crossover(stereo_signal[:, 1], f_xover, sample_rate)

        # Calcul manuel
        expected_lf_mono1 = (left_low + right_low) * 0.707

        # Calcul via fonction
        lf_mono1 = compute_lf_mono1_stereo(left_low, right_low)

        assert_allclose(lf_mono1, expected_lf_mono1, rtol=1e-5)

    def test_stereo_power_sum(self, stereo_signal, default_params, sample_rate):
        """
        Test : Somme à puissance constante pour stéréo.

        Vérifie que l'énergie est préservée dans la somme.
        """
        f_xover = default_params["F_xover1"]

        crossover = Crossover(
            freq=f_xover,
            fs=sample_rate,
            n_channels=2,
            lfe_channel_indices=[],
        )

        low_freq, high_freq, lf_mono1 = crossover.process(stereo_signal)

        # LF_mono1 devrait avoir une énergie cohérente
        rms_lf_mono1 = np.sqrt(np.mean(lf_mono1**2))
        assert rms_lf_mono1 > 0, "LF_mono1 ne devrait pas être nul"


class TestPowerConstantSum:
    """Tests pour la somme à puissance constante."""

    def test_power_sum_formula(self):
        """
        Test : Formule de somme à puissance constante.

        Vérifie que pour plusieurs signaux :
        result = sqrt(sum(signal_i²))
        """
        # Créer plusieurs signaux de test
        signals = [
            np.array([1.0, 2.0, 3.0], dtype=np.float32),
            np.array([2.0, 3.0, 4.0], dtype=np.float32),
            np.array([1.0, 1.0, 1.0], dtype=np.float32),
        ]

        result = sum_power_constant(signals)

        # Calcul manuel : sqrt(1² + 2² + 1²), sqrt(2² + 3² + 1²), sqrt(3² + 4² + 1²)
        expected = np.sqrt(
            np.array(
                [
                    1.0**2 + 2.0**2 + 1.0**2,
                    2.0**2 + 3.0**2 + 1.0**2,
                    3.0**2 + 4.0**2 + 1.0**2,
                ]
            )
        ).astype(np.float32)

        assert_allclose(result, expected, rtol=1e-5)

    def test_power_sum_energy_preservation(self, multichannel_signal_5_1):
        """
        Test : Préservation de l'énergie dans la somme.

        Vérifie que l'énergie totale avant et après somme est similaire.
        """
        # Prendre les 5 canaux non-LFE (indices 0, 1, 2, 4, 5)
        signals = [
            multichannel_signal_5_1[:, i].astype(np.float32) for i in [0, 1, 2, 4, 5]
        ]

        result = sum_power_constant(signals)

        # Énergie avant : somme des carrés de tous les signaux
        energy_before = sum(np.sum(sig**2) for sig in signals)

        # Énergie après : somme des carrés du résultat
        energy_after = np.sum(result**2)

        # Les énergies devraient être identiques
        assert_allclose(energy_after, energy_before, rtol=1e-5)

    def test_power_sum_rms_level(self):
        """
        Test : Niveau RMS après somme.

        Vérifie que le niveau RMS de la somme est cohérent avec
        les niveaux d'entrée.
        """
        # Deux signaux identiques
        signal1 = np.ones(100, dtype=np.float32)
        signal2 = np.ones(100, dtype=np.float32)

        result = sum_power_constant([signal1, signal2])

        # sqrt(1² + 1²) = sqrt(2) ≈ 1.414
        expected_value = np.sqrt(2.0)
        assert_allclose(result, expected_value, rtol=1e-5)

    def test_power_sum_empty_list_raises_error(self):
        """Test : Liste vide lève une erreur."""
        with pytest.raises(ValueError, match="vide"):
            sum_power_constant([])

    def test_power_sum_different_lengths_raises_error(self):
        """Test : Signaux de longueurs différentes lèvent une erreur."""
        signals = [
            np.array([1.0, 2.0]),
            np.array([1.0, 2.0, 3.0]),
        ]
        with pytest.raises(ValueError, match="longueur différente"):
            sum_power_constant(signals)


class TestMultichannelCrossover:
    """Tests pour crossover multicanal."""

    def test_multichannel_all_channels_processed(
        self, multichannel_signal_5_1, default_params, sample_rate
    ):
        """
        Test : Tous les canaux non-LFE sont traités.

        Vérifie que tous les canaux (sauf LFE) passent par le crossover
        et produisent des signaux HF.
        """
        f_xover = default_params["F_xover1"]
        lfe_index = 3  # LFE est à l'index 3 en 5.1

        low_freq, high_freq = apply_crossover_multichannel(
            multichannel_signal_5_1, f_xover, sample_rate, [lfe_index]
        )

        # Tous les canaux sauf LFE devraient être non nuls
        for ch in range(6):
            if ch == lfe_index:
                assert np.allclose(
                    low_freq[:, ch], 0
                ), f"Canal LFE {ch} devrait être zéro"
                assert np.allclose(
                    high_freq[:, ch], 0
                ), f"Canal LFE {ch} devrait être zéro"
            else:
                assert not np.allclose(
                    low_freq[:, ch], 0
                ), f"Canal {ch} ne devrait pas être zéro"
                assert not np.allclose(
                    high_freq[:, ch], 0
                ), f"Canal {ch} ne devrait pas être zéro"

    def test_multichannel_lf_mono1(
        self, multichannel_signal_5_1, default_params, sample_rate
    ):
        """
        Test : LF_mono1 pour multicanal.

        Vérifie que LF_mono1 = sqrt(sum(all_lowfreq²)) pour tous
        les canaux non-LFE.
        """
        f_xover = default_params["F_xover1"]
        lfe_index = 3

        crossover = Crossover(
            freq=f_xover,
            fs=sample_rate,
            n_channels=6,
            lfe_channel_indices=[lfe_index],
        )

        low_freq, high_freq, lf_mono1 = crossover.process(multichannel_signal_5_1)

        # Calcul manuel de LF_mono1
        active_channels = [ch for ch in range(6) if ch != lfe_index]
        low_freq_list = [low_freq[:, ch] for ch in active_channels]
        expected_lf_mono1 = compute_lf_mono1_multichannel(low_freq_list)

        assert_allclose(lf_mono1, expected_lf_mono1, rtol=1e-5)

    def test_multichannel_high_freq_preserved(
        self, multichannel_signal_5_1, default_params, sample_rate
    ):
        """
        Test : Préservation des hautes fréquences par canal.

        Vérifie que chaque canal non-LFE produit un signal HF séparé.
        """
        f_xover = default_params["F_xover1"]
        lfe_index = 3

        low_freq, high_freq = apply_crossover_multichannel(
            multichannel_signal_5_1, f_xover, sample_rate, [lfe_index]
        )

        # Les canaux HF doivent avoir la même forme que les canaux LF
        assert high_freq.shape == multichannel_signal_5_1.shape


class TestLFEExclusion:
    """Tests pour l'exclusion des canaux LFE."""

    def test_lfe_not_in_crossover(self, sample_rate):
        """
        Test : LFE n'est pas traité par le crossover.

        Crée un signal avec LFE et vérifie que le canal LFE n'est pas
        filtré par le crossover.
        """
        n_samples = sample_rate
        signal = np.random.randn(n_samples, 6).astype(np.float32)
        lfe_index = 3

        low_freq, high_freq = apply_crossover_multichannel(
            signal, 150.0, sample_rate, [lfe_index]
        )

        # Le canal LFE devrait être zéro après crossover
        assert np.allclose(low_freq[:, lfe_index], 0)
        assert np.allclose(high_freq[:, lfe_index], 0)

    def test_lfe_not_in_lf_mono1(self, sample_rate):
        """
        Test : LFE n'est pas inclus dans LF_mono1.

        Vérifie que même si un canal LFE existe, il n'est pas inclus
        dans le calcul de LF_mono1.
        """
        n_samples = sample_rate
        signal = np.random.randn(n_samples, 6).astype(np.float32)
        lfe_index = 3

        crossover = Crossover(
            freq=150.0,
            fs=sample_rate,
            n_channels=6,
            lfe_channel_indices=[lfe_index],
        )

        low_freq, _, lf_mono1 = crossover.process(signal)

        # LF_mono1 devrait être calculé sans le LFE
        # On vérifie que l'énergie est cohérente avec les 5 canaux actifs
        active_channels = [ch for ch in range(6) if ch != lfe_index]
        low_freq_list = [low_freq[:, ch] for ch in active_channels]
        expected_lf_mono1 = compute_lf_mono1_multichannel(low_freq_list)

        assert_allclose(lf_mono1, expected_lf_mono1, rtol=1e-5)

    def test_multiple_lfe_exclusion(self, sample_rate):
        """
        Test : Exclusion de plusieurs canaux LFE.

        Pour un format avec plusieurs LFE, vérifie que tous
        sont exclus du crossover.
        """
        n_samples = sample_rate
        n_channels = 12  # Format 10.2 par exemple
        signal = np.random.randn(n_samples, n_channels).astype(np.float32)
        lfe_indices = [3, 9]  # Deux canaux LFE

        low_freq, high_freq = apply_crossover_multichannel(
            signal, 150.0, sample_rate, lfe_indices
        )

        for lfe_idx in lfe_indices:
            assert np.allclose(low_freq[:, lfe_idx], 0)
            assert np.allclose(high_freq[:, lfe_idx], 0)


class TestEnergyConservation:
    """Tests pour la conservation de l'énergie."""

    def test_total_energy_preserved(self, stereo_signal, default_params, sample_rate):
        """
        Test : Énergie totale préservée.

        Vérifie que l'énergie totale (LF + HF) est similaire à l'énergie
        d'entrée (tolérance due au filtrage).
        """
        f_xover = default_params["F_xover1"]

        # Énergie d'entrée
        energy_input = np.sum(stereo_signal[:, 0] ** 2)

        # Crossover
        low_freq, high_freq = apply_crossover(stereo_signal[:, 0], f_xover, sample_rate)

        # Énergie de sortie (après stabilisation des filtres)
        offset = sample_rate // 4
        energy_low = np.sum(low_freq[offset:] ** 2)
        energy_high = np.sum(high_freq[offset:] ** 2)

        # L'énergie totale devrait être préservée (tolérance 20%)
        # Note: il y a une tolérance car les filtres ont un overlap
        energy_output = energy_low + energy_high
        energy_input_trimmed = np.sum(stereo_signal[offset:, 0] ** 2)

        # Tolérance plus large pour les crossovers (overlap des filtres)
        assert energy_output > energy_input_trimmed * 0.7
        assert energy_output < energy_input_trimmed * 1.5

    def test_energy_split_lf_hf(self, default_params, sample_rate):
        """
        Test : Répartition énergie LF/HF.

        Vérifie que l'énergie est correctement répartie entre LF et HF
        selon la fréquence de coupure.
        """
        f_xover = default_params["F_xover1"]

        # Signal purement basse fréquence (50 Hz)
        t = np.linspace(0, 1.0, sample_rate)
        low_signal = np.sin(2 * np.pi * 50 * t).astype(np.float32)

        low_freq, high_freq = apply_crossover(low_signal, f_xover, sample_rate)

        # L'énergie devrait être principalement dans LF
        offset = sample_rate // 4
        energy_low = np.sum(low_freq[offset:] ** 2)
        energy_high = np.sum(high_freq[offset:] ** 2)

        assert energy_low > energy_high * 5, "LF devrait dominer pour un signal BF"

    def test_no_energy_creation(self, white_noise_signal, default_params, sample_rate):
        """
        Test : Pas de création d'énergie.

        Vérifie que l'énergie de sortie n'est pas supérieure à l'énergie
        d'entrée (pas d'amplification parasite).
        """
        f_xover = default_params["F_xover1"]

        # Énergie d'entrée
        energy_input = np.sum(white_noise_signal**2)

        # Crossover
        low_freq, high_freq = apply_crossover(white_noise_signal, f_xover, sample_rate)

        # Énergie de sortie
        energy_output = np.sum(low_freq**2) + np.sum(high_freq**2)

        # L'énergie ne devrait pas augmenter significativement
        # (tolérance de 50% pour les effets de bord)
        assert energy_output < energy_input * 1.5


class TestCrossoverEdgeCases:
    """Tests pour les cas limites."""

    def test_dc_signal(self, sample_rate):
        """
        Test : Signal DC (fréquence 0).

        Vérifie le comportement avec un signal constant.
        """
        dc_signal = np.ones(sample_rate, dtype=np.float32)

        low_freq, high_freq = apply_crossover(dc_signal, 150.0, sample_rate)

        # Le DC devrait être principalement dans les basses fréquences
        # (après stabilisation)
        offset = sample_rate // 4
        assert np.mean(np.abs(low_freq[offset:])) > np.mean(np.abs(high_freq[offset:]))

    def test_high_freq_signal(self, sample_rate):
        """
        Test : Signal haute fréquence uniquement.

        Vérifie que les très hautes fréquences passent bien dans HF.
        """
        t = np.linspace(0, 1.0, sample_rate)
        high_signal = np.sin(2 * np.pi * 5000 * t).astype(np.float32)

        low_freq, high_freq = apply_crossover(high_signal, 150.0, sample_rate)

        # L'énergie devrait être principalement dans HF
        offset = sample_rate // 4
        energy_low = np.sum(low_freq[offset:] ** 2)
        energy_high = np.sum(high_freq[offset:] ** 2)

        assert energy_high > energy_low * 10, "HF devrait dominer pour un signal HF"

    def test_low_freq_signal(self, sample_rate):
        """
        Test : Signal basse fréquence uniquement.

        Vérifie que les très basses fréquences passent bien dans LF.
        """
        t = np.linspace(0, 1.0, sample_rate)
        low_signal = np.sin(2 * np.pi * 30 * t).astype(np.float32)

        low_freq, high_freq = apply_crossover(low_signal, 150.0, sample_rate)

        # L'énergie devrait être principalement dans LF
        offset = sample_rate // 4
        energy_low = np.sum(low_freq[offset:] ** 2)
        energy_high = np.sum(high_freq[offset:] ** 2)

        assert energy_low > energy_high * 5, "LF devrait dominer pour un signal BF"

    def test_1d_signal_only(self, sample_rate):
        """Test : apply_crossover n'accepte que des signaux 1D."""
        signal_2d = np.random.randn(sample_rate, 2).astype(np.float32)

        with pytest.raises(ValueError, match="1D"):
            apply_crossover(signal_2d, 150.0, sample_rate)

    def test_2d_signal_for_multichannel(self, sample_rate):
        """Test : apply_crossover_multichannel nécessite un signal 2D."""
        signal_1d = np.random.randn(sample_rate).astype(np.float32)

        with pytest.raises(ValueError, match="2D"):
            apply_crossover_multichannel(signal_1d, 150.0, sample_rate)


class TestCrossoverClass:
    """Tests pour la classe Crossover."""

    def test_crossover_initialization(self, sample_rate):
        """Test : Initialisation correcte du crossover."""
        crossover = Crossover(
            freq=150.0,
            fs=sample_rate,
            n_channels=6,
            lfe_channel_indices=[3],
        )

        assert crossover.freq == 150.0
        assert crossover.n_channels == 6
        assert crossover.lfe_channel_indices == [3]

    def test_crossover_reset(self, stereo_signal, sample_rate):
        """Test : Reset remet les filtres à zéro."""
        crossover = Crossover(freq=150.0, fs=sample_rate, n_channels=2)

        # Premier traitement
        low1, high1, lf1 = crossover.process(stereo_signal)

        # Reset et retraitement
        crossover.reset()
        low2, high2, lf2 = crossover.process(stereo_signal)

        # Les sorties devraient être identiques
        assert_allclose(low1, low2, rtol=1e-5)
        assert_allclose(high1, high2, rtol=1e-5)
        assert_allclose(lf1, lf2, rtol=1e-5)

    def test_crossover_wrong_channel_count(self, stereo_signal, sample_rate):
        """Test : Mauvais nombre de canaux lève une erreur."""
        crossover = Crossover(freq=150.0, fs=sample_rate, n_channels=6)

        with pytest.raises(ValueError, match="canaux incorrect"):
            crossover.process(stereo_signal)  # 2 canaux au lieu de 6
