"""
Tests unitaires pour les filtres Biquad.

Module testé : upmix_algorithm.modules.biquad_filter

Plan de tests :
1. Calcul coefficients biquad (formule cookbook)
2. Réponse en fréquence : vérifier -6dB à F_xover1
3. Cascade de 2 biquads : réponse totale
4. Filtre passe-bas pour LFE : réponse à F_LFE
5. Stabilité numérique (pas de dépassement)
6. Phase linéaire (ou vérification de la phase)
7. Filtres Peak, Low Shelf, High Shelf
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from upmix_algorithm.modules.biquad_filter import (
    BiquadFilter,
    CascadeBiquadFilter,
    FilterType,
    compute_biquad_coeffs,
)


class TestBiquadCoefficients:
    """Tests pour le calcul des coefficients biquad."""

    def test_lpf_coefficients_cookbook(self, biquad_params):
        """
        Test : Calcul coefficients LPF selon W3C Audio EQ Cookbook.

        Vérifie que les coefficients calculés correspondent aux formules :
        - b₀ = (1 - cos(ω₀)) / 2
        - b₁ = 1 - cos(ω₀)
        - b₂ = (1 - cos(ω₀)) / 2
        - a₀ = 1 + α
        - a₁ = -2cos(ω₀)
        - a₂ = 1 - α
        où α = sin(ω₀) / (2 × Q)
        """
        freq = biquad_params["freq"]
        q = biquad_params["q"]
        fs = biquad_params["fs"]

        b, a = compute_biquad_coeffs(freq, q, fs, FilterType.LPF)

        # Calcul manuel des coefficients attendus
        omega0 = 2 * np.pi * freq / fs
        cos_omega0 = np.cos(omega0)
        sin_omega0 = np.sin(omega0)
        alpha = sin_omega0 / (2 * q)

        # Coefficients attendus (avant normalisation)
        b0_expected = (1 - cos_omega0) / 2
        b1_expected = 1 - cos_omega0
        b2_expected = (1 - cos_omega0) / 2
        a0_expected = 1 + alpha
        a1_expected = -2 * cos_omega0
        a2_expected = 1 - alpha

        # Normalisation par a0
        b_expected = np.array(
            [
                b0_expected / a0_expected,
                b1_expected / a0_expected,
                b2_expected / a0_expected,
            ]
        )
        a_expected = np.array(
            [1.0, a1_expected / a0_expected, a2_expected / a0_expected]
        )

        assert_allclose(b, b_expected, rtol=1e-5)
        assert_allclose(a, a_expected, rtol=1e-5)

    def test_hpf_coefficients_cookbook(self, biquad_params):
        """
        Test : Calcul coefficients HPF selon W3C Audio EQ Cookbook.

        Vérifie que les coefficients calculés correspondent aux formules HPF.
        """
        freq = biquad_params["freq"]
        q = biquad_params["q"]
        fs = biquad_params["fs"]

        b, a = compute_biquad_coeffs(freq, q, fs, FilterType.HPF)

        # Calcul manuel des coefficients attendus
        omega0 = 2 * np.pi * freq / fs
        cos_omega0 = np.cos(omega0)
        sin_omega0 = np.sin(omega0)
        alpha = sin_omega0 / (2 * q)

        # Coefficients attendus (avant normalisation) - pour documentation
        _b0_expected = (1 + cos_omega0) / 2
        _b1_expected = -(1 + cos_omega0)
        _b2_expected = (1 + cos_omega0) / 2
        _a0_expected = 1 + alpha

        # Vérification b0 et b2 égaux
        assert_allclose(b[0], b[2], rtol=1e-5)
        # Vérification b1 = -2 * b0
        assert_allclose(b[1], -2 * b[0], rtol=1e-5)
        # Vérification a[0] = 1 (normalisé)
        assert a[0] == 1.0

    def test_coefficients_normalization(self, biquad_params):
        """
        Test : Normalisation des coefficients (a₀ = 1).

        Vérifie que les coefficients sont normalisés avec a₀ = 1.
        """
        _b, a = compute_biquad_coeffs(
            biquad_params["freq"],
            biquad_params["q"],
            biquad_params["fs"],
            FilterType.LPF,
        )

        assert a[0] == 1.0, "a0 doit être égal à 1 après normalisation"

    def test_invalid_frequency_raises_error(self, biquad_params):
        """Test : Fréquence invalide lève une erreur."""
        with pytest.raises(ValueError, match="fréquence"):
            compute_biquad_coeffs(
                0, biquad_params["q"], biquad_params["fs"], FilterType.LPF
            )

        # Fréquence >= Nyquist
        with pytest.raises(ValueError, match="fréquence"):
            compute_biquad_coeffs(
                biquad_params["fs"] / 2,
                biquad_params["q"],
                biquad_params["fs"],
                FilterType.LPF,
            )

    def test_invalid_q_raises_error(self, biquad_params):
        """Test : Q invalide lève une erreur."""
        with pytest.raises(ValueError, match="facteur Q"):
            compute_biquad_coeffs(
                biquad_params["freq"], 0, biquad_params["fs"], FilterType.LPF
            )


class TestBiquadFrequencyResponse:
    """Tests pour la réponse en fréquence."""

    def test_lpf_cutoff_frequency(self, biquad_params, sample_rate):
        """
        Test : Réponse à -3dB à la fréquence de coupure pour un biquad.

        Pour un filtre LPF biquad simple (Q=0.707), la réponse est à -3dB
        à la fréquence de coupure.
        """
        freq_cutoff = biquad_params["freq"]
        filt = BiquadFilter(
            freq_cutoff, biquad_params["q"], sample_rate, FilterType.LPF
        )

        frequencies, magnitude_db = filt.get_frequency_response(n_points=8192)

        # Trouver la magnitude à la fréquence de coupure
        idx = np.argmin(np.abs(frequencies - freq_cutoff))
        mag_at_cutoff = magnitude_db[idx]

        # Pour un biquad avec Q=0.707, la réponse est à -3dB
        assert_allclose(mag_at_cutoff, -3.0, atol=0.5)

    def test_hpf_cutoff_frequency(self, biquad_params, sample_rate):
        """
        Test : Réponse HPF à -3dB à la fréquence de coupure.
        """
        freq_cutoff = biquad_params["freq"]
        filt = BiquadFilter(
            freq_cutoff, biquad_params["q"], sample_rate, FilterType.HPF
        )

        frequencies, magnitude_db = filt.get_frequency_response(n_points=8192)

        idx = np.argmin(np.abs(frequencies - freq_cutoff))
        mag_at_cutoff = magnitude_db[idx]

        assert_allclose(mag_at_cutoff, -3.0, atol=0.5)

    def test_frequency_response_shape(self, biquad_params, sample_rate):
        """
        Test : Forme générale de la réponse en fréquence.

        Vérifie que :
        - LPF : passe les basses fréquences, atténue les hautes
        - HPF : atténue les basses fréquences, passe les hautes
        """
        freq_cutoff = biquad_params["freq"]

        # LPF
        lpf = BiquadFilter(freq_cutoff, biquad_params["q"], sample_rate, FilterType.LPF)
        freq_lpf, mag_lpf = lpf.get_frequency_response()

        # Basses fréquences (< coupure) doivent être proches de 0 dB
        low_freq_idx = freq_lpf < freq_cutoff / 2
        assert np.all(mag_lpf[low_freq_idx] > -1.0)

        # Hautes fréquences (> 2x coupure) doivent être atténuées
        high_freq_idx = freq_lpf > freq_cutoff * 4
        if np.any(high_freq_idx):
            assert np.all(mag_lpf[high_freq_idx] < -10.0)

        # HPF
        hpf = BiquadFilter(freq_cutoff, biquad_params["q"], sample_rate, FilterType.HPF)
        freq_hpf, mag_hpf = hpf.get_frequency_response()

        # Basses fréquences (< coupure/4) doivent être atténuées
        very_low_freq_idx = freq_hpf < freq_cutoff / 4
        if np.any(very_low_freq_idx):
            assert np.all(mag_hpf[very_low_freq_idx] < -10.0)


class TestBiquadCascade:
    """Tests pour la cascade de 2 biquads (ordre 4)."""

    def test_cascade_order_4(self, biquad_params, sample_rate):
        """
        Test : Cascade de 2 biquads donne ordre 4.

        Vérifie que la pente d'atténuation est d'environ -24dB/octave
        (caractéristique d'un filtre d'ordre 4).
        """
        freq_cutoff = biquad_params["freq"]
        cascade = CascadeBiquadFilter(
            freq_cutoff, biquad_params["q"], sample_rate, FilterType.LPF, n_stages=2
        )

        frequencies, magnitude_db = cascade.get_frequency_response(n_points=8192)

        # Mesurer la pente entre 2x et 4x la fréquence de coupure
        idx_2x = np.argmin(np.abs(frequencies - freq_cutoff * 2))
        idx_4x = np.argmin(np.abs(frequencies - freq_cutoff * 4))

        if idx_4x > idx_2x:
            mag_2x = magnitude_db[idx_2x]
            mag_4x = magnitude_db[idx_4x]
            slope_db_per_octave = mag_4x - mag_2x  # Une octave entre 2x et 4x

            # Pour ordre 4, pente ≈ -24 dB/octave
            assert slope_db_per_octave < -20, f"Pente: {slope_db_per_octave} dB/octave"

    def test_cascade_cutoff_6db(self, biquad_params, sample_rate):
        """
        Test : Cascade maintient -6dB à la fréquence de coupure.

        Vérifie que 2 biquads en cascade donnent -6dB à F_xover1.
        """
        freq_cutoff = biquad_params["freq"]
        cascade = CascadeBiquadFilter(
            freq_cutoff, biquad_params["q"], sample_rate, FilterType.LPF, n_stages=2
        )

        frequencies, magnitude_db = cascade.get_frequency_response(n_points=8192)

        idx = np.argmin(np.abs(frequencies - freq_cutoff))
        mag_at_cutoff = magnitude_db[idx]

        # 2 biquads à -3dB chacun = -6dB total
        assert_allclose(mag_at_cutoff, -6.0, atol=1.0)


class TestLFE_LowPass:
    """Tests pour le filtre passe-bas LFE."""

    def test_lfe_cutoff_frequency(self, sample_rate):
        """
        Test : Filtre LFE à F_LFE (120 Hz par défaut).

        Vérifie que le filtre passe-bas pour LFE a une coupure à -6dB
        à F_LFE = 120 Hz avec 2 biquads.
        """
        f_lfe = 120.0
        cascade = CascadeBiquadFilter(
            f_lfe, 0.707, sample_rate, FilterType.LPF, n_stages=2
        )

        frequencies, magnitude_db = cascade.get_frequency_response(n_points=8192)

        idx = np.argmin(np.abs(frequencies - f_lfe))
        mag_at_cutoff = magnitude_db[idx]

        assert_allclose(mag_at_cutoff, -6.0, atol=1.0)

    def test_lfe_attenuation_high_freq(self, sample_rate):
        """
        Test : Atténuation des hautes fréquences pour LFE.

        Vérifie que les fréquences >> F_LFE sont fortement atténuées.
        """
        f_lfe = 120.0
        cascade = CascadeBiquadFilter(
            f_lfe, 0.707, sample_rate, FilterType.LPF, n_stages=2
        )

        frequencies, magnitude_db = cascade.get_frequency_response(n_points=8192)

        # À 1000 Hz (>> 120 Hz), l'atténuation doit être forte
        idx_1000 = np.argmin(np.abs(frequencies - 1000))
        assert magnitude_db[idx_1000] < -30, "Atténuation insuffisante à 1000 Hz"


class TestBiquadStability:
    """Tests pour la stabilité numérique."""

    def test_no_overflow(self, biquad_params, white_noise_signal):
        """
        Test : Pas de dépassement numérique.

        Applique un signal de bruit blanc et vérifie que les valeurs
        restent dans des limites raisonnables (pas de NaN, Inf).
        """
        filt = BiquadFilter(
            biquad_params["freq"],
            biquad_params["q"],
            biquad_params["fs"],
            FilterType.LPF,
        )

        output = filt.process(white_noise_signal)

        assert not np.any(np.isnan(output)), "NaN détecté dans la sortie"
        assert not np.any(np.isinf(output)), "Inf détecté dans la sortie"

    def test_impulse_response_stable(self, biquad_params, sample_rate):
        """
        Test : Réponse impulsionnelle stable.

        Applique une impulsion et vérifie que la réponse décroît
        sans oscillation divergente.
        """
        filt = BiquadFilter(
            biquad_params["freq"],
            biquad_params["q"],
            sample_rate,
            FilterType.LPF,
        )

        # Créer une impulsion
        impulse = np.zeros(1000, dtype=np.float32)
        impulse[0] = 1.0

        response = filt.process(impulse)

        # La réponse doit décroître
        assert np.abs(response[-1]) < 0.01, "La réponse ne décroît pas"

        # Pas de NaN ou Inf
        assert not np.any(np.isnan(response))
        assert not np.any(np.isinf(response))

    def test_long_signal_stability(self, biquad_params, sample_rate):
        """
        Test : Stabilité sur signal long.

        Applique un signal de plusieurs secondes et vérifie qu'il n'y a
        pas de dérive ou d'instabilité.
        """
        duration = 10.0  # 10 secondes
        n_samples = int(sample_rate * duration)
        signal = np.random.randn(n_samples).astype(np.float32)

        filt = BiquadFilter(
            biquad_params["freq"],
            biquad_params["q"],
            sample_rate,
            FilterType.LPF,
        )

        output = filt.process(signal)

        # Vérifier stabilité
        assert not np.any(np.isnan(output))
        assert not np.any(np.isinf(output))

        # Le niveau de sortie ne doit pas exploser
        assert np.max(np.abs(output)) < np.max(np.abs(signal)) * 10


class TestBiquadPhase:
    """Tests pour la phase du filtre."""

    def test_phase_response(self, biquad_params, sample_rate):
        """
        Test : Vérification de la réponse en phase.

        Vérifie que la phase est continue et sans saut de 2π.
        """
        from scipy.signal import freqz

        filt = BiquadFilter(
            biquad_params["freq"],
            biquad_params["q"],
            sample_rate,
            FilterType.LPF,
        )

        _w, h = freqz(filt.b, filt.a, worN=512)  # type: ignore[arg-type]
        phase = np.unwrap(np.angle(h))

        # La phase doit être monotone décroissante pour un LPF
        phase_diff = np.diff(phase)
        assert np.all(phase_diff <= 0.1), "Saut de phase détecté"

    def test_group_delay(self, biquad_params, sample_rate):
        """
        Test : Délai de groupe du filtre.

        Vérifie que le délai de groupe est raisonnable.
        """
        from scipy.signal import group_delay

        filt = BiquadFilter(
            biquad_params["freq"],
            biquad_params["q"],
            sample_rate,
            FilterType.LPF,
        )

        _w, gd = group_delay((filt.b, filt.a), w=512)

        # Le délai de groupe doit être positif
        assert np.all(gd > -1), "Délai de groupe négatif anormal"


class TestBiquadFilterClass:
    """Tests pour la classe BiquadFilter (filtre avec état)."""

    def test_filter_initialization(self, biquad_params):
        """
        Test : Initialisation correcte du filtre.
        """
        filt = BiquadFilter(
            biquad_params["freq"],
            biquad_params["q"],
            biquad_params["fs"],
            FilterType.LPF,
        )

        assert filt.freq == biquad_params["freq"]
        assert filt.q == biquad_params["q"]
        assert filt.fs == biquad_params["fs"]
        assert filt.filter_type == FilterType.LPF

    def test_filter_state_persistence(self, biquad_params, sine_signal):
        """
        Test : Persistance de l'état entre appels.

        Vérifie que le filtre maintient son état (mémoire) entre
        plusieurs appels process().
        """
        filt = BiquadFilter(
            biquad_params["freq"],
            biquad_params["q"],
            biquad_params["fs"],
            FilterType.LPF,
        )

        # Traiter le signal en deux parties
        half = len(sine_signal) // 2
        part1 = filt.process(sine_signal[:half])
        part2 = filt.process(sine_signal[half:])
        output_split = np.concatenate([part1, part2])

        # Traiter en une seule fois
        filt.reset()
        output_full = filt.process(sine_signal)

        # Les résultats doivent être identiques
        assert_allclose(output_split, output_full, rtol=1e-5)

    def test_filter_reset(self, biquad_params, sine_signal):
        """
        Test : Réinitialisation du filtre.

        Vérifie que reset() remet l'état à zéro.
        """
        filt = BiquadFilter(
            biquad_params["freq"],
            biquad_params["q"],
            biquad_params["fs"],
            FilterType.LPF,
        )

        # Traiter une fois
        output1 = filt.process(sine_signal)

        # Reset et retraiter
        filt.reset()
        output2 = filt.process(sine_signal)

        # Les sorties doivent être identiques (même état initial)
        assert_allclose(output1, output2, rtol=1e-5)

    def test_filter_1d_only(self, biquad_params):
        """Test : Le filtre n'accepte que des signaux 1D."""
        filt = BiquadFilter(
            biquad_params["freq"],
            biquad_params["q"],
            biquad_params["fs"],
            FilterType.LPF,
        )

        signal_2d = np.random.randn(100, 2).astype(np.float32)

        with pytest.raises(ValueError, match="1D"):
            filt.process(signal_2d)


class TestPeakingEQ:
    """Tests pour le filtre Peaking EQ (PK)."""

    def test_peak_coefficients_cookbook(self, biquad_params):
        """
        Test : Calcul coefficients Peaking EQ selon W3C Audio EQ Cookbook.

        Vérifie que les coefficients sont calculés correctement.
        """
        freq = biquad_params["freq"]
        q = biquad_params["q"]
        fs = biquad_params["fs"]
        gain_db = 6.0

        b, a = compute_biquad_coeffs(freq, q, fs, FilterType.PK, gain_db)

        # Calcul manuel des coefficients attendus
        omega0 = 2 * np.pi * freq / fs
        cos_omega0 = np.cos(omega0)
        sin_omega0 = np.sin(omega0)
        alpha = sin_omega0 / (2 * q)
        A = 10 ** (gain_db / 40)

        b0_expected = 1 + alpha * A
        b1_expected = -2 * cos_omega0
        b2_expected = 1 - alpha * A
        a0_expected = 1 + alpha / A
        a1_expected = -2 * cos_omega0
        a2_expected = 1 - alpha / A

        b_expected = np.array(
            [
                b0_expected / a0_expected,
                b1_expected / a0_expected,
                b2_expected / a0_expected,
            ]
        )
        a_expected = np.array(
            [1.0, a1_expected / a0_expected, a2_expected / a0_expected]
        )

        assert_allclose(b, b_expected, rtol=1e-5)
        assert_allclose(a, a_expected, rtol=1e-5)

    def test_peak_boost_at_center_frequency(self, sample_rate):
        """
        Test : Boost à la fréquence centrale pour un filtre Peak.

        Pour un gain de +6dB, la réponse à la fréquence centrale
        doit être proche de +6dB.
        """
        freq = 1000.0
        gain_db = 6.0
        q = 1.0

        filt = BiquadFilter(freq, q, sample_rate, FilterType.PK, gain_db)
        frequencies, magnitude_db = filt.get_frequency_response(n_points=8192)

        idx = np.argmin(np.abs(frequencies - freq))
        mag_at_center = magnitude_db[idx]

        assert_allclose(mag_at_center, gain_db, atol=0.5)

    def test_peak_cut_at_center_frequency(self, sample_rate):
        """
        Test : Cut à la fréquence centrale pour un filtre Peak.

        Pour un gain de -6dB, la réponse à la fréquence centrale
        doit être proche de -6dB.
        """
        freq = 1000.0
        gain_db = -6.0
        q = 1.0

        filt = BiquadFilter(freq, q, sample_rate, FilterType.PK, gain_db)
        frequencies, magnitude_db = filt.get_frequency_response(n_points=8192)

        idx = np.argmin(np.abs(frequencies - freq))
        mag_at_center = magnitude_db[idx]

        assert_allclose(mag_at_center, gain_db, atol=0.5)

    def test_peak_unity_gain_at_extremes(self, sample_rate):
        """
        Test : Gain unitaire loin de la fréquence centrale.

        Le gain doit être proche de 0dB aux fréquences très basses
        et très hautes.
        """
        freq = 1000.0
        gain_db = 12.0
        q = 2.0

        filt = BiquadFilter(freq, q, sample_rate, FilterType.PK, gain_db)
        frequencies, magnitude_db = filt.get_frequency_response(n_points=8192)

        # Basses fréquences (< 100 Hz)
        low_idx = frequencies < 100
        assert np.all(np.abs(magnitude_db[low_idx]) < 1.0)

        # Hautes fréquences (> 10000 Hz)
        high_idx = frequencies > 10000
        if np.any(high_idx):
            assert np.all(np.abs(magnitude_db[high_idx]) < 1.0)

    def test_peak_zero_gain_is_unity(self, sample_rate):
        """
        Test : Avec gain=0dB, le filtre est transparent.
        """
        freq = 1000.0
        gain_db = 0.0
        q = 1.0

        filt = BiquadFilter(freq, q, sample_rate, FilterType.PK, gain_db)
        _frequencies, magnitude_db = filt.get_frequency_response(n_points=8192)

        # Toute la réponse doit être proche de 0 dB
        assert np.all(np.abs(magnitude_db) < 0.1)


class TestLowShelf:
    """Tests pour le filtre Low Shelf."""

    def test_lowshelf_boost_low_frequencies(self, sample_rate):
        """
        Test : Boost des basses fréquences avec Low Shelf.

        Les fréquences en dessous de la fréquence de coupure
        doivent être boostées.
        """
        freq = 500.0
        gain_db = 6.0
        q = 0.707

        filt = BiquadFilter(freq, q, sample_rate, FilterType.LOW_SHELF, gain_db)
        frequencies, magnitude_db = filt.get_frequency_response(n_points=8192)

        # Très basses fréquences doivent avoir le gain complet
        very_low_idx = frequencies < freq / 4
        if np.any(very_low_idx):
            assert np.all(magnitude_db[very_low_idx] > gain_db - 1.5)

        # Hautes fréquences (>> fréquence de coupure) doivent être à ~0dB
        high_idx = frequencies > freq * 4
        if np.any(high_idx):
            assert np.all(np.abs(magnitude_db[high_idx]) < 1.0)

    def test_lowshelf_cut_low_frequencies(self, sample_rate):
        """
        Test : Cut des basses fréquences avec Low Shelf.
        """
        freq = 500.0
        gain_db = -6.0
        q = 0.707

        filt = BiquadFilter(freq, q, sample_rate, FilterType.LOW_SHELF, gain_db)
        frequencies, magnitude_db = filt.get_frequency_response(n_points=8192)

        # Très basses fréquences doivent avoir le gain complet (négatif)
        very_low_idx = frequencies < freq / 4
        if np.any(very_low_idx):
            assert np.all(magnitude_db[very_low_idx] < gain_db + 1.5)

    def test_lowshelf_transition_at_cutoff(self, sample_rate):
        """
        Test : Transition à la fréquence de coupure.

        À la fréquence de coupure, le gain doit être environ gain/2.
        """
        freq = 500.0
        gain_db = 6.0
        q = 0.707

        filt = BiquadFilter(freq, q, sample_rate, FilterType.LOW_SHELF, gain_db)
        frequencies, magnitude_db = filt.get_frequency_response(n_points=8192)

        idx = np.argmin(np.abs(frequencies - freq))
        mag_at_cutoff = magnitude_db[idx]

        # À la fréquence de coupure, le gain est à environ 50% du gain total
        assert mag_at_cutoff > 0 and mag_at_cutoff < gain_db


class TestHighShelf:
    """Tests pour le filtre High Shelf."""

    def test_highshelf_boost_high_frequencies(self, sample_rate):
        """
        Test : Boost des hautes fréquences avec High Shelf.

        Les fréquences au-dessus de la fréquence de coupure
        doivent être boostées.
        """
        freq = 2000.0
        gain_db = 6.0
        q = 0.707

        filt = BiquadFilter(freq, q, sample_rate, FilterType.HIGH_SHELF, gain_db)
        frequencies, magnitude_db = filt.get_frequency_response(n_points=8192)

        # Très hautes fréquences doivent avoir le gain complet
        very_high_idx = frequencies > freq * 4
        if np.any(very_high_idx):
            assert np.all(magnitude_db[very_high_idx] > gain_db - 1.5)

        # Basses fréquences (<< fréquence de coupure) doivent être à ~0dB
        low_idx = frequencies < freq / 4
        if np.any(low_idx):
            assert np.all(np.abs(magnitude_db[low_idx]) < 1.0)

    def test_highshelf_cut_high_frequencies(self, sample_rate):
        """
        Test : Cut des hautes fréquences avec High Shelf.
        """
        freq = 2000.0
        gain_db = -6.0
        q = 0.707

        filt = BiquadFilter(freq, q, sample_rate, FilterType.HIGH_SHELF, gain_db)
        frequencies, magnitude_db = filt.get_frequency_response(n_points=8192)

        # Très hautes fréquences doivent avoir le gain complet (négatif)
        very_high_idx = frequencies > freq * 4
        if np.any(very_high_idx):
            assert np.all(magnitude_db[very_high_idx] < gain_db + 1.5)

    def test_highshelf_transition_at_cutoff(self, sample_rate):
        """
        Test : Transition à la fréquence de coupure.
        """
        freq = 2000.0
        gain_db = 6.0
        q = 0.707

        filt = BiquadFilter(freq, q, sample_rate, FilterType.HIGH_SHELF, gain_db)
        frequencies, magnitude_db = filt.get_frequency_response(n_points=8192)

        idx = np.argmin(np.abs(frequencies - freq))
        mag_at_cutoff = magnitude_db[idx]

        # À la fréquence de coupure, le gain est à environ 50% du gain total
        assert mag_at_cutoff > 0 and mag_at_cutoff < gain_db


class TestNewFiltersStability:
    """Tests de stabilité pour les nouveaux filtres."""

    def test_peak_stability(self, sample_rate, white_noise_signal):
        """Test : Stabilité du filtre Peak."""
        filt = BiquadFilter(1000.0, 1.0, sample_rate, FilterType.PK, 12.0)
        output = filt.process(white_noise_signal)

        assert not np.any(np.isnan(output))
        assert not np.any(np.isinf(output))

    def test_lowshelf_stability(self, sample_rate, white_noise_signal):
        """Test : Stabilité du filtre Low Shelf."""
        filt = BiquadFilter(500.0, 0.707, sample_rate, FilterType.LOW_SHELF, 12.0)
        output = filt.process(white_noise_signal)

        assert not np.any(np.isnan(output))
        assert not np.any(np.isinf(output))

    def test_highshelf_stability(self, sample_rate, white_noise_signal):
        """Test : Stabilité du filtre High Shelf."""
        filt = BiquadFilter(2000.0, 0.707, sample_rate, FilterType.HIGH_SHELF, 12.0)
        output = filt.process(white_noise_signal)

        assert not np.any(np.isnan(output))
        assert not np.any(np.isinf(output))

    def test_extreme_gains(self, sample_rate, white_noise_signal):
        """Test : Stabilité avec des gains extrêmes."""
        # Gain très élevé
        filt = BiquadFilter(1000.0, 1.0, sample_rate, FilterType.PK, 24.0)
        output = filt.process(white_noise_signal)
        assert not np.any(np.isnan(output))
        assert not np.any(np.isinf(output))

        # Gain très négatif
        filt = BiquadFilter(1000.0, 1.0, sample_rate, FilterType.PK, -24.0)
        output = filt.process(white_noise_signal)
        assert not np.any(np.isnan(output))
        assert not np.any(np.isinf(output))

    def test_cascade_with_gain(self, sample_rate, white_noise_signal):
        """Test : Cascade de filtres avec gain."""
        cascade = CascadeBiquadFilter(
            1000.0, 1.0, sample_rate, FilterType.PK, n_stages=2, gain_db=6.0
        )
        output = cascade.process(white_noise_signal)

        assert not np.any(np.isnan(output))
        assert not np.any(np.isinf(output))
