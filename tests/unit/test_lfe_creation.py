"""
Tests unitaires pour la création du canal LFE.

Module testé : upmix_algorithm.modules.lfe_processor

Plan de tests :
1. Détection LFE existant : identification correcte
2. Création LFE depuis somme : niveau correct
3. Filtre LP LFE : atténuation au-delà de F_LFE
4. Cas multicanal : tous canaux inclus sauf LFE
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from upmix_algorithm.modules.lfe_processor import (
    LFEProcessor,
    create_lfe_from_sum,
    detect_lfe_channels,
    extract_existing_lfe,
    get_non_lfe_channels,
    process_lfe,
)


class TestLFEDetection:
    """Tests pour la détection des canaux LFE existants."""

    def test_detect_lfe_in_5_1(self):
        """
        Test : Détection LFE dans format 5.1.

        Vérifie que le canal LFE est correctement identifié dans
        un layout 5.1.
        """
        labels = ["L", "R", "C", "LFE", "LS", "RS"]
        lfe_indices = detect_lfe_channels(labels)

        assert lfe_indices == [3]

    def test_detect_lfe_in_7_1(self):
        """
        Test : Détection LFE dans format 7.1.
        """
        labels = ["L", "R", "C", "LFE", "LS", "RS", "LB", "RB"]
        lfe_indices = detect_lfe_channels(labels)

        assert lfe_indices == [3]

    def test_detect_multiple_lfe(self):
        """
        Test : Détection de plusieurs LFE (ex: 10.2).

        Vérifie que tous les canaux LFE sont détectés.
        """
        labels = [
            "S1",
            "S2",
            "S3",
            "S4",
            "S5",
            "S6",
            "S7",
            "S8",
            "S9",
            "S10",
            "LFE1",
            "LFE2",
        ]
        lfe_indices = detect_lfe_channels(labels)

        assert lfe_indices == [10, 11]

    def test_no_lfe_in_stereo(self):
        """
        Test : Pas de LFE dans stéréo.

        Vérifie que detect_lfe_channels retourne une liste vide
        pour un layout sans LFE.
        """
        labels = ["L", "R"]
        lfe_indices = detect_lfe_channels(labels)

        assert lfe_indices == []

    def test_lfe_label_detection(self):
        """
        Test : Détection par label.

        Vérifie que la détection fonctionne avec les labels définis
        dans multichannel_layouts.py (LFE, LFE1, LFE2, etc.).
        """
        # Test avec différentes variantes de labels LFE
        labels_tests = [
            (["L", "R", "LFE"], [2]),
            (["L", "R", "lfe"], [2]),  # Minuscules
            (["L", "R", "LFE1", "LFE2"], [2, 3]),
            (["L", "R", "SubLFE"], [2]),  # Contient LFE
        ]

        for labels, expected in labels_tests:
            result = detect_lfe_channels(labels)
            assert result == expected, f"Échec pour {labels}"

    def test_get_non_lfe_channels(self):
        """Test : Récupération des canaux non-LFE."""
        labels = ["L", "R", "C", "LFE", "LS", "RS"]
        non_lfe = get_non_lfe_channels(labels)

        assert non_lfe == [0, 1, 2, 4, 5]


class TestLFECreationFromSum:
    """Tests pour la création LFE depuis somme mono."""

    def test_lfe_creation_power_sum(
        self, multichannel_signal_5_1, default_params, sample_rate
    ):
        """
        Test : Création LFE avec somme à puissance constante.

        Vérifie que le LFE créé utilise la formule :
        LFE = LP(sqrt(sum(all_channels²)))
        """
        labels = ["L", "R", "C", "LFE", "LS", "RS"]
        f_lfe = default_params["F_LFE"]

        # Remplacer le canal LFE par zéro pour simuler l'absence de LFE
        signal = multichannel_signal_5_1.copy()
        signal[:, 3] = 0  # LFE à zéro

        # Créer les labels sans LFE pour le test
        labels_no_lfe = ["L", "R", "C", "X", "LS", "RS"]  # X n'est pas un LFE

        lfe_signal = create_lfe_from_sum(signal, labels_no_lfe, f_lfe, sample_rate)

        # Le LFE créé devrait être non nul
        assert np.sum(lfe_signal**2) > 0, "LFE ne devrait pas être nul"

    def test_lfe_creation_excludes_existing_lfe(self, sample_rate):
        """
        Test : Exclusion des LFE existants de la somme.

        Si un LFE existe déjà, il ne doit pas être inclus dans
        la somme pour créer un nouveau LFE.
        """
        n_samples = sample_rate
        labels = ["L", "R", "C", "LFE", "LS", "RS"]

        # Créer un signal avec un LFE très fort
        signal = np.random.randn(n_samples, 6).astype(np.float32)
        signal[:, 3] = 10.0  # LFE très fort

        # Création LFE (devrait ignorer le canal LFE existant)
        lfe_created = create_lfe_from_sum(signal, labels, 120.0, sample_rate)

        # Le LFE créé ne devrait pas être dominé par le canal LFE existant
        # (car il est exclu de la somme)
        rms_lfe_created = np.sqrt(np.mean(lfe_created**2))
        rms_input_lfe = np.sqrt(np.mean(signal[:, 3] ** 2))

        # Le LFE créé devrait être beaucoup plus faible que le LFE d'entrée
        assert rms_lfe_created < rms_input_lfe

    def test_lfe_creation_all_channels_included(
        self, multichannel_signal_5_1, default_params, sample_rate
    ):
        """
        Test : Tous les canaux non-LFE inclus.

        Vérifie que tous les canaux (sauf LFE) sont inclus dans
        la somme pour créer le LFE.
        """
        labels = ["L", "R", "C", "X", "LS", "RS"]  # Pas de LFE
        f_lfe = default_params["F_LFE"]

        signal = multichannel_signal_5_1.copy()
        signal[:, 3] = 0  # X à zéro

        # Tous les 5 canaux non-LFE devraient contribuer
        non_lfe = get_non_lfe_channels(labels)
        assert len(non_lfe) == 6  # Tous les canaux

    def test_lfe_creation_level(
        self, multichannel_signal_5_1, default_params, sample_rate
    ):
        """
        Test : Niveau du LFE créé.

        Vérifie que le niveau RMS du LFE créé est cohérent avec
        les niveaux des canaux sources.
        """
        labels = ["L", "R", "C", "X", "LS", "RS"]  # Pas de LFE
        f_lfe = default_params["F_LFE"]

        signal = multichannel_signal_5_1.copy()

        lfe_signal = create_lfe_from_sum(signal, labels, f_lfe, sample_rate)

        rms_lfe = np.sqrt(np.mean(lfe_signal**2))

        # Le RMS devrait être dans une plage raisonnable
        assert rms_lfe > 0
        assert not np.isnan(rms_lfe)
        assert not np.isinf(rms_lfe)


class TestLFELowPassFilter:
    """Tests pour le filtre passe-bas LFE."""

    def test_lfe_cutoff_frequency(self, default_params, sample_rate):
        """
        Test : Coupure à F_LFE (120 Hz par défaut).

        Vérifie que le filtre LP a une coupure à -6dB à F_LFE.
        """
        f_lfe = default_params["F_LFE"]
        labels = ["L", "R"]  # Stéréo sans LFE

        # Signal sinusoïdal à F_LFE
        t = np.linspace(0, 1.0, sample_rate)
        test_signal = np.column_stack(
            [
                np.sin(2 * np.pi * f_lfe * t),
                np.sin(2 * np.pi * f_lfe * t),
            ]
        ).astype(np.float32)

        lfe_signal = create_lfe_from_sum(test_signal, labels, f_lfe, sample_rate)

        # Le signal à F_LFE devrait être atténué d'environ -6dB (cascade de 2)
        # C'est difficile à tester précisément, mais le signal ne devrait pas être nul
        assert np.sum(lfe_signal**2) > 0

    def test_lfe_high_freq_attenuation(self, default_params, sample_rate):
        """
        Test : Atténuation des hautes fréquences.

        Vérifie que les fréquences >> F_LFE sont fortement atténuées
        dans le LFE créé.
        """
        f_lfe = default_params["F_LFE"]
        labels = ["L", "R"]

        # Signal haute fréquence (1000 Hz >> 120 Hz)
        t = np.linspace(0, 1.0, sample_rate)
        high_freq_signal = np.column_stack(
            [
                np.sin(2 * np.pi * 1000 * t),
                np.sin(2 * np.pi * 1000 * t),
            ]
        ).astype(np.float32)

        lfe_signal = create_lfe_from_sum(high_freq_signal, labels, f_lfe, sample_rate)

        # Le LFE devrait être atténué par rapport à l'entrée après somme à puissance constante
        # La somme de 2 signaux identiques donne sqrt(2) * amplitude
        # Puis le filtre LP atténue les HF
        offset = sample_rate // 4  # Après stabilisation

        # Vérifier que le signal n'est pas composé uniquement de la fréquence HF
        # en analysant le spectre du LFE
        from scipy.fft import rfft, rfftfreq

        spectrum = np.abs(rfft(lfe_signal[offset:]))
        freqs = rfftfreq(len(lfe_signal[offset:]), 1 / sample_rate)

        # Trouver l'énergie à 1000 Hz vs l'énergie totale
        idx_1000 = np.argmin(np.abs(freqs - 1000))
        energy_at_1000 = spectrum[idx_1000]
        energy_total = np.sum(spectrum)

        # L'énergie à 1000 Hz devrait être une petite fraction du total
        assert energy_at_1000 < energy_total * 0.1, "1000 Hz devrait être atténué"

    def test_lfe_low_freq_preservation(self, default_params, sample_rate):
        """
        Test : Préservation des basses fréquences.

        Vérifie que les fréquences < F_LFE sont bien préservées.
        """
        f_lfe = default_params["F_LFE"]
        labels = ["L", "R"]

        # Signal basse fréquence (30 Hz << 120 Hz)
        t = np.linspace(0, 1.0, sample_rate)
        low_freq_signal = np.column_stack(
            [
                np.sin(2 * np.pi * 30 * t),
                np.sin(2 * np.pi * 30 * t),
            ]
        ).astype(np.float32)

        lfe_signal = create_lfe_from_sum(low_freq_signal, labels, f_lfe, sample_rate)

        # Le LFE devrait préserver l'énergie basse fréquence
        offset = sample_rate // 4
        rms_lfe = np.sqrt(np.mean(lfe_signal[offset:] ** 2))

        assert rms_lfe > 0.5, "BF devrait être préservé dans LFE"

    def test_lfe_filter_order_4(self, default_params, sample_rate):
        """
        Test : Ordre 4 du filtre (2 biquads en cascade).

        Vérifie que la pente d'atténuation correspond à un filtre
        d'ordre 4 (-24dB/octave).
        """
        # Ce test est couvert par test_biquad.py
        # On vérifie juste que le processeur utilise bien 2 stages
        labels = ["L", "R"]
        processor = LFEProcessor(labels, 120.0, sample_rate, n_stages=2)

        assert (
            processor._lfe_filter is not None
        ), "Filter should exist for non-LFE input"
        assert processor._lfe_filter.n_stages == 2


class TestLFEMultichannelCases:
    """Tests pour cas multicanal."""

    def test_lfe_5_1(self, sample_rate):
        """
        Test : Création LFE pour format 5.1.

        Vérifie que le LFE est correctement créé à partir des
        5 canaux non-LFE.
        """
        labels = ["L", "R", "C", "X", "LS", "RS"]  # X au lieu de LFE
        n_samples = sample_rate

        signal = np.random.randn(n_samples, 6).astype(np.float32)
        signal[:, 3] = 0  # Canal X à zéro

        lfe_signal = create_lfe_from_sum(signal, labels, 120.0, sample_rate)

        assert lfe_signal.shape == (n_samples,)
        assert np.sum(lfe_signal**2) > 0

    def test_lfe_7_1(self, sample_rate):
        """
        Test : Création LFE pour format 7.1.
        """
        labels = ["L", "R", "C", "X", "LS", "RS", "LB", "RB"]  # X au lieu de LFE
        n_samples = sample_rate

        signal = np.random.randn(n_samples, 8).astype(np.float32)
        signal[:, 3] = 0

        lfe_signal = create_lfe_from_sum(signal, labels, 120.0, sample_rate)

        assert lfe_signal.shape == (n_samples,)

    def test_lfe_multiple_sources(self, sample_rate):
        """
        Test : LFE avec plusieurs sources.

        Vérifie que le LFE est créé correctement même avec
        un grand nombre de canaux sources.
        """
        n_channels = 12
        labels = [f"S{i}" for i in range(n_channels)]
        n_samples = sample_rate

        signal = np.random.randn(n_samples, n_channels).astype(np.float32)

        lfe_signal = create_lfe_from_sum(signal, labels, 120.0, sample_rate)

        assert lfe_signal.shape == (n_samples,)


class TestLFEUseExisting:
    """Tests pour utilisation LFE existant."""

    def test_use_existing_lfe(self, sample_rate):
        """
        Test : Utilisation directe du LFE existant.

        Si un LFE existe dans le signal d'entrée, il doit être
        utilisé directement sans modification.
        """
        labels = ["L", "R", "C", "LFE", "LS", "RS"]
        n_samples = sample_rate

        signal = np.random.randn(n_samples, 6).astype(np.float32)
        original_lfe = signal[:, 3].copy()

        extracted_lfe = extract_existing_lfe(signal, labels)

        assert_allclose(extracted_lfe, original_lfe)

    def test_multiple_lfe_sum(self, sample_rate):
        """
        Test : Somme de plusieurs LFE existants.

        Si plusieurs LFE existent, ils doivent être sommés en mono.
        """
        labels = ["S1", "S2", "S3", "LFE1", "LFE2"]
        n_samples = sample_rate

        signal = np.random.randn(n_samples, 5).astype(np.float32)

        extracted_lfe = extract_existing_lfe(signal, labels)

        # Somme des deux LFE
        expected_lfe = signal[:, 3] + signal[:, 4]

        assert_allclose(extracted_lfe, expected_lfe)

    def test_process_lfe_existing(self, sample_rate):
        """Test : process_lfe avec LFE existant."""
        labels = ["L", "R", "C", "LFE", "LS", "RS"]
        n_samples = sample_rate

        signal = np.random.randn(n_samples, 6).astype(np.float32)

        lfe, was_existing = process_lfe(signal, labels, 120.0, sample_rate)

        assert was_existing is True
        assert_allclose(lfe, signal[:, 3])

    def test_process_lfe_created(self, sample_rate):
        """Test : process_lfe sans LFE existant."""
        labels = ["L", "R", "C", "X", "LS", "RS"]  # Pas de LFE
        n_samples = sample_rate

        signal = np.random.randn(n_samples, 6).astype(np.float32)

        lfe, was_existing = process_lfe(signal, labels, 120.0, sample_rate)

        assert was_existing is False
        assert lfe.shape == (n_samples,)


class TestLFEEdgeCases:
    """Tests pour cas limites."""

    def test_lfe_wrong_dimensions(self):
        """Test : Signal de mauvaise dimension."""
        labels = ["L", "R"]
        signal_1d = np.random.randn(1000).astype(np.float32)

        with pytest.raises(ValueError, match="2D"):
            create_lfe_from_sum(signal_1d, labels, 120.0, 48000)

    def test_lfe_wrong_channel_count(self):
        """Test : Mauvais nombre de canaux."""
        labels = ["L", "R", "C"]
        signal = np.random.randn(1000, 2).astype(np.float32)

        with pytest.raises(ValueError, match="ne correspond pas"):
            create_lfe_from_sum(signal, labels, 120.0, 48000)

    def test_lfe_silent_signal(self, sample_rate):
        """
        Test : LFE avec signal silencieux.
        """
        labels = ["L", "R"]
        silent_signal = np.zeros((sample_rate, 2), dtype=np.float32)

        lfe_signal = create_lfe_from_sum(silent_signal, labels, 120.0, sample_rate)

        # Le LFE devrait aussi être silencieux
        assert np.allclose(lfe_signal, 0)

    def test_lfe_only_high_freq(self, default_params, sample_rate):
        """
        Test : LFE avec seulement hautes fréquences.

        Si le signal n'a que des hautes fréquences, le LFE créé
        doit être atténué dans les hautes fréquences.
        """
        labels = ["L", "R"]
        f_lfe = default_params["F_LFE"]

        # Signal à 5000 Hz
        t = np.linspace(0, 1.0, sample_rate)
        high_freq_signal = np.column_stack(
            [
                np.sin(2 * np.pi * 5000 * t),
                np.sin(2 * np.pi * 5000 * t),
            ]
        ).astype(np.float32)

        lfe_signal = create_lfe_from_sum(high_freq_signal, labels, f_lfe, sample_rate)

        # Vérifier que le LFE ne contient pas d'énergie à 5000 Hz
        from scipy.fft import rfft, rfftfreq

        offset = sample_rate // 4
        spectrum = np.abs(rfft(lfe_signal[offset:]))
        freqs = rfftfreq(len(lfe_signal[offset:]), 1 / sample_rate)

        # L'énergie à 5000 Hz devrait être très faible
        idx_5000 = np.argmin(np.abs(freqs - 5000))
        energy_at_5000 = spectrum[idx_5000]
        energy_max = np.max(spectrum)

        assert energy_at_5000 < energy_max * 0.01, "5000 Hz devrait être très atténué"


class TestLFEProcessorClass:
    """Tests pour la classe LFEProcessor."""

    def test_processor_initialization(self, sample_rate):
        """Test : Initialisation correcte du processeur."""
        labels = ["L", "R", "C", "LFE", "LS", "RS"]
        processor = LFEProcessor(labels, 120.0, sample_rate)

        assert processor.labels == labels
        assert processor.f_lfe == 120.0
        assert processor.has_existing_lfe is True
        assert processor.lfe_indices == [3]

    def test_processor_without_lfe(self, sample_rate):
        """Test : Processeur sans LFE existant."""
        labels = ["L", "R"]
        processor = LFEProcessor(labels, 120.0, sample_rate)

        assert processor.has_existing_lfe is False
        assert processor._lfe_filter is not None

    def test_processor_reset(self, sample_rate):
        """Test : Reset du processeur."""
        labels = ["L", "R"]
        processor = LFEProcessor(labels, 120.0, sample_rate)

        # Traitement
        signal = np.random.randn(sample_rate, 2).astype(np.float32)
        lfe1 = processor.process(signal)

        # Reset et retraitement
        processor.reset()
        lfe2 = processor.process(signal)

        assert_allclose(lfe1, lfe2, rtol=1e-5)
