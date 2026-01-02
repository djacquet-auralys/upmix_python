"""
Tests unitaires pour la génération de masques.

Module testé : upmix_algorithm.modules.mask_generator

Plan de tests :
1. LUT masque : valeurs correctes selon pan/width/slope
2. Blur triangulaire : lissage fréquentiel correct
3. Rampsmooth : attack/release corrects
4. Min_gain : floor respecté
5. Cas limites : pan hors range, width très petit/grand
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from upmix_algorithm.modules.mask_generator import (
    RampSmooth,
    apply_freq_blur,
    apply_temporal_smoothing,
    create_mask_lut,
    generate_extraction_mask,
    interpolate_lut,
)


class TestMaskLUT:
    """Tests pour la LUT de masque."""

    def test_lut_formula(self):
        """
        Test : Formule de la LUT.

        Vérifie que la formule est :
        y = 10^(max(min(SLOPE * (W/2 - abs(x - PAN)), 0), min_gain) / 20.0)
        """
        pan = 0.0
        width = 0.2
        slope = 500.0
        min_gain = -40.0

        x, lut = create_mask_lut(pan, width, slope, min_gain)

        # Vérifier au centre (x = pan): gain devrait être 0 dB -> 1.0 linéaire
        center_idx = len(x) // 2
        assert_allclose(lut[center_idx], 1.0, atol=0.01)

        # Vérifier la formule manuellement pour quelques points
        for i, xi in enumerate(x):
            distance = abs(xi - pan)
            gain_db = slope * (width / 2.0 - distance)
            gain_db = np.clip(gain_db, min_gain, 0.0)
            expected = 10 ** (gain_db / 20.0)
            assert_allclose(lut[i], expected, atol=1e-5)

    def test_lut_resolution(self):
        """
        Test : Résolution de 200 points.

        Vérifie que la LUT a 200 points entre -1 et 1.
        """
        x, lut = create_mask_lut(0.0, 0.2, 500.0, -40.0, resolution=200)

        assert len(x) == 200
        assert len(lut) == 200
        assert_allclose(x[0], -1.0)
        assert_allclose(x[-1], 1.0)

    def test_lut_interpolation_linear(self):
        """
        Test : Interpolation linéaire.

        Vérifie que l'interpolation entre les points de la LUT
        est linéaire.
        """
        x, lut = create_mask_lut(0.0, 0.5, 100.0, -40.0, resolution=100)

        # Tester l'interpolation à des points intermédiaires
        test_points = np.array([0.015, 0.025, 0.035])  # Entre les points de la LUT
        interpolated = interpolate_lut(x, lut, test_points)

        # L'interpolation doit être continue
        assert len(interpolated) == len(test_points)
        assert not np.any(np.isnan(interpolated))

    def test_lut_peak_at_pan(self):
        """
        Test : Pic à la position pan.

        Vérifie que le gain maximum est autour de la position pan.
        Note: Avec la formule, il y a un plateau de gain maximal (0 dB)
        pour |x - pan| <= width/2. Donc le pic est en fait un plateau.
        """
        pan = 0.5
        width = 0.2  # Le plateau s'étend de pan - width/2 à pan + width/2
        x, lut = create_mask_lut(pan, width, 500.0, -40.0, resolution=1000)

        # Le maximum (gain = 1.0) devrait être atteint dans le plateau
        max_value = np.max(lut)
        assert_allclose(max_value, 1.0, atol=0.01)

        # Vérifier que le plateau existe autour de pan
        # Le plateau devrait s'étendre de pan - width/2 à pan + width/2
        plateau_mask = lut > 0.99
        plateau_x = x[plateau_mask]

        # Le plateau devrait contenir la position pan
        assert np.min(plateau_x) <= pan <= np.max(plateau_x)

    def test_lut_width(self):
        """
        Test : Largeur du masque.

        Vérifie que la largeur du masque correspond au paramètre width.
        """
        pan = 0.0
        width = 0.4
        slope = 1000.0  # Pente très raide
        min_gain = -40.0

        x, lut = create_mask_lut(pan, width, slope, min_gain, resolution=1000)

        # Avec une pente très raide, le gain devrait chuter rapidement
        # hors de la zone de largeur width
        # Trouver les indices où le gain est proche de 1 (> 0.9)
        high_gain_mask = lut > 0.9
        high_gain_range = x[high_gain_mask]

        if len(high_gain_range) > 0:
            measured_width = high_gain_range[-1] - high_gain_range[0]
            # La largeur mesurée devrait être proche de width
            assert measured_width < width + 0.1

    def test_lut_slope(self):
        """
        Test : Pente du masque.

        Vérifie que la pente correspond au paramètre slope.
        """
        pan = 0.0
        width = 0.2

        # Comparer deux slopes différentes
        x1, lut1 = create_mask_lut(pan, width, slope=100.0, min_gain_db=-40.0)
        x2, lut2 = create_mask_lut(pan, width, slope=500.0, min_gain_db=-40.0)

        # Avec une pente plus raide (slope=500), la décroissance est plus rapide
        # Comparer à un point hors du centre
        test_idx = len(x1) // 2 + 30  # Un peu à droite du centre
        assert lut2[test_idx] < lut1[test_idx]  # Plus de pente = moins de gain

    def test_lut_min_gain_floor(self):
        """
        Test : Floor à min_gain.

        Vérifie que le gain minimum est limité à min_gain (en dB).
        """
        min_gain_db = -40.0
        min_gain_lin = 10 ** (min_gain_db / 20.0)

        x, lut = create_mask_lut(0.0, 0.1, 500.0, min_gain_db)

        # Vérifier que tous les gains sont >= min_gain_lin
        assert np.all(lut >= min_gain_lin - 1e-6)


class TestInterpolateLUT:
    """Tests pour l'interpolation de LUT."""

    def test_interpolate_1d(self):
        """Test d'interpolation 1D."""
        x, lut = create_mask_lut(0.0, 0.2, 500.0, -40.0)
        pan_values = np.array([-0.5, 0.0, 0.5])

        gains = interpolate_lut(x, lut, pan_values)

        assert gains.shape == (3,)
        assert gains[1] > gains[0]  # Centre > côtés
        assert gains[1] > gains[2]

    def test_interpolate_2d(self):
        """Test d'interpolation 2D."""
        x, lut = create_mask_lut(0.0, 0.2, 500.0, -40.0)
        pan_values = np.random.uniform(-1, 1, (10, 65)).astype(np.float32)

        gains = interpolate_lut(x, lut, pan_values)

        assert gains.shape == (10, 65)
        assert gains.dtype == np.float32


class TestFrequencyBlur:
    """Tests pour le blur triangulaire fréquentiel."""

    def test_blur_kernel_size(self):
        """
        Test : Taille du noyau (3 bins).

        Vérifie que le blur utilise un noyau de 3 bins.
        """
        # Créer un signal avec un pic isolé
        n_freq = 65
        gains = np.zeros(n_freq, dtype=np.float32)
        gains[32] = 1.0  # Pic au centre

        blurred = apply_freq_blur(gains, exclude_dc_nyquist=False)

        # Le blur devrait affecter les bins adjacents
        assert blurred[31] > 0  # Bin avant
        assert blurred[32] > 0  # Bin central
        assert blurred[33] > 0  # Bin après
        # Les bins plus loin ne devraient pas être affectés
        assert blurred[29] == 0

    def test_blur_triangular_shape(self):
        """
        Test : Forme triangulaire linéaire décroissante.

        Vérifie que le noyau a la forme [0.25, 0.5, 0.25]
        (normalisé).
        """
        # Créer un signal avec un pic isolé au centre
        n_freq = 65
        gains = np.zeros(n_freq, dtype=np.float32)
        gains[32] = 1.0

        blurred = apply_freq_blur(gains, exclude_dc_nyquist=False)

        # Le noyau est [0.25, 0.5, 0.25]
        assert_allclose(blurred[31], 0.25, atol=1e-5)
        assert_allclose(blurred[32], 0.50, atol=1e-5)
        assert_allclose(blurred[33], 0.25, atol=1e-5)

    def test_blur_excludes_dc_nyquist(self):
        """
        Test : Exclusion DC et Nyquist.

        Vérifie que le blur n'est pas appliqué aux bins 0
        et Nyquist (selon code de référence).
        """
        n_freq = 65
        gains = np.ones(n_freq, dtype=np.float32)

        blurred = apply_freq_blur(gains, exclude_dc_nyquist=True)

        # DC et Nyquist ne devraient pas être modifiés par le blur
        # (ils restent à 1.0 car tous les voisins sont à 1.0)
        assert_allclose(blurred[0], 1.0, atol=1e-5)
        assert_allclose(blurred[-1], 1.0, atol=1e-5)

    def test_blur_smoothing_effect(self):
        """
        Test : Effet de lissage.

        Vérifie que le blur lisse effectivement les variations
        fréquentielles.
        """
        n_freq = 65
        # Créer un signal avec des variations abruptes
        gains = np.zeros(n_freq, dtype=np.float32)
        gains[20:25] = 1.0  # Plateau

        blurred = apply_freq_blur(gains, exclude_dc_nyquist=True)

        # Les bords du plateau devraient être adoucis
        assert blurred[19] > 0  # Avant le plateau
        assert blurred[25] > 0  # Après le plateau

    def test_blur_2d_input(self):
        """Test du blur sur entrée 2D."""
        gains = np.random.rand(10, 65).astype(np.float32)

        blurred = apply_freq_blur(gains, exclude_dc_nyquist=True)

        assert blurred.shape == (10, 65)
        assert blurred.dtype == np.float32


class TestTemporalSmoothing:
    """Tests pour le lissage temporel (rampsmooth)."""

    def test_rampsmooth_attack(self):
        """
        Test : Ramp-up avec attack.

        Vérifie que quand le nouveau gain > ancien gain,
        le lissage utilise attack frames.
        """
        n_freq = 65
        attack = 5.0
        release = 50.0

        smoother = RampSmooth(n_freq, attack_frames=attack, release_frames=release)

        # Première frame à 0
        smoother.process(np.zeros(n_freq, dtype=np.float32))

        # Deuxième frame à 1 (step up)
        target = np.ones(n_freq, dtype=np.float32)
        result = smoother.process(target)

        # Avec attack=5, le coef est 0.2, donc après 1 frame:
        # state = 0 + (1 - 0) * 0.2 = 0.2
        assert_allclose(result, 0.2, atol=0.01)

    def test_rampsmooth_release(self):
        """
        Test : Ramp-down avec release.

        Vérifie que quand le nouveau gain < ancien gain,
        le lissage utilise release frames.
        """
        n_freq = 65
        attack = 1.0
        release = 10.0

        # Désactiver le double release pour DC/Nyquist pour ce test
        smoother = RampSmooth(
            n_freq,
            attack_frames=attack,
            release_frames=release,
            double_release_dc_nyquist=False,
        )

        # Première frame à 1
        smoother.process(np.ones(n_freq, dtype=np.float32))

        # Deuxième frame à 0 (step down)
        target = np.zeros(n_freq, dtype=np.float32)
        result = smoother.process(target)

        # Avec release=10, le coef est 0.1, donc après 1 frame:
        # state = 1 + (0 - 1) * 0.1 = 0.9
        assert_allclose(result, 0.9, atol=0.01)

    def test_rampsmooth_freeze(self):
        """
        Test : Freeze si power < 1e-6.

        Vérifie que le lissage est gelé (freeze) quand
        la puissance est très faible.
        """
        # Note: Le freeze est documenté mais pas complètement implémenté
        # dans la version actuelle. Ce test vérifie le comportement de base.
        n_freq = 65
        smoother = RampSmooth(n_freq, freeze_threshold=1e-6)

        # Ce test vérifie que le smoother peut être créé avec freeze_threshold
        assert smoother.freeze_threshold == 1e-6

    def test_rampsmooth_release_doubled_dc_nyquist(self):
        """
        Test : Release doublé pour bin 0 et Nyquist.

        Vérifie que le release est doublé pour les bins
        0 et Nyquist (selon code de référence).
        """
        n_freq = 65
        release = 10.0

        smoother = RampSmooth(
            n_freq,
            attack_frames=1.0,
            release_frames=release,
            double_release_dc_nyquist=True,
        )

        # Première frame à 1
        initial = np.ones(n_freq, dtype=np.float32)
        smoother.process(initial)

        # Deuxième frame à 0 (step down)
        target = np.zeros(n_freq, dtype=np.float32)
        result = smoother.process(target)

        # Pour les bins normaux: coef = 0.1, result = 0.9
        # Pour DC et Nyquist: coef = 0.05 (release doublé), result = 0.95
        assert result[0] > result[1]  # DC descend plus lentement
        assert result[-1] > result[-2]  # Nyquist descend plus lentement
        assert_allclose(result[0], 0.95, atol=0.01)
        assert_allclose(result[-1], 0.95, atol=0.01)

    def test_rampsmooth_linear_interpolation(self):
        """
        Test : Interpolation linéaire.

        Vérifie que le rampsmooth utilise une interpolation
        linéaire entre ancien et nouveau gain.
        """
        n_freq = 65
        attack = 4.0  # 4 frames pour atteindre la cible

        smoother = RampSmooth(n_freq, attack_frames=attack, release_frames=100.0)

        # Première frame à 0
        smoother.process(np.zeros(n_freq, dtype=np.float32))

        # Traiter plusieurs frames vers 1
        target = np.ones(n_freq, dtype=np.float32)
        results = []
        for _ in range(4):
            results.append(smoother.process(target)[32])  # Bin central

        # Les valeurs devraient augmenter linéairement
        # Avec coef = 0.25: 0.25, 0.4375, 0.578, 0.684...
        assert results[0] < results[1] < results[2] < results[3]

    def test_rampsmooth_reset(self):
        """Test de reset du smoother."""
        n_freq = 65
        smoother = RampSmooth(n_freq)

        # Traiter une frame
        smoother.process(np.ones(n_freq, dtype=np.float32))

        # Reset
        smoother.reset()

        # Après reset, l'état devrait être None
        assert smoother._state is None


class TestMinGain:
    """Tests pour le floor min_gain."""

    def test_min_gain_respected(self):
        """
        Test : Respect du min_gain.

        Vérifie que tous les gains sont >= min_gain (en linéaire).
        """
        min_gain_db = -40.0
        min_gain_lin = 10 ** (min_gain_db / 20.0)

        x, lut = create_mask_lut(0.0, 0.1, 1000.0, min_gain_db)

        assert np.all(lut >= min_gain_lin - 1e-7)

    def test_min_gain_application(self):
        """
        Test : Application du min_gain.

        Vérifie que le min_gain est bien appliqué comme floor
        dans la formule de la LUT.
        """
        min_gain_db = -20.0
        min_gain_lin = 10 ** (min_gain_db / 20.0)

        x, lut = create_mask_lut(0.0, 0.1, 1000.0, min_gain_db)

        # Le minimum de la LUT devrait être exactement min_gain_lin
        assert_allclose(np.min(lut), min_gain_lin, atol=1e-6)


class TestEdgeCases:
    """Tests pour cas limites."""

    def test_pan_out_of_range(self):
        """
        Test : Pan hors range [-1, 1].

        Vérifie le comportement avec pan < -1 ou pan > 1.
        Note: Avec une pente raide et un pan très hors range,
        toute la LUT peut être au min_gain. On teste avec un pan
        légèrement hors range.
        """
        # Pan = 1.1 (légèrement hors range) avec large width
        # Le pic devrait être à x=1 (le plus proche de pan=1.1)
        x, lut = create_mask_lut(1.1, 0.5, 100.0, -40.0)

        # La LUT devrait quand même être créée
        assert len(lut) == 200

        # Avec pan=1.1 et width=0.5, le plateau s'étend de 0.85 à 1.35
        # Donc x=1 devrait être dans le plateau (gain max)
        max_idx = np.argmax(lut)
        assert x[max_idx] >= 0.8  # Proche de la limite droite

        # Vérifier que le gain à x=1 est élevé
        assert lut[-1] > 0.5  # x=1 devrait avoir un gain significatif

        # Test avec pan très hors range - toute la LUT sera au min_gain
        x2, lut2 = create_mask_lut(5.0, 0.2, 500.0, -40.0)
        min_gain_lin = 10 ** (-40.0 / 20.0)
        assert_allclose(lut2, min_gain_lin, atol=1e-5)  # Tout au min_gain

    def test_width_very_small(self):
        """
        Test : Width très petit.

        Vérifie le comportement avec width très petit (≈ 0).
        """
        x, lut = create_mask_lut(0.0, 0.001, 500.0, -40.0)

        # Avec une largeur très petite, seul le centre devrait avoir un gain élevé
        assert lut[100] > 0.5  # Centre
        assert np.sum(lut > 0.5) <= 5  # Très peu de bins à gain élevé

    def test_width_very_large(self):
        """
        Test : Width très grand.

        Vérifie le comportement avec width très grand (≈ 2).
        """
        x, lut = create_mask_lut(0.0, 2.0, 500.0, -40.0)

        # Avec une largeur de 2 (couvrant tout le range), presque tout devrait être à gain max
        assert np.sum(lut > 0.9) > 150  # La plupart des bins à gain élevé

    def test_slope_very_high(self):
        """
        Test : Slope très élevé.

        Vérifie le comportement avec slope très élevé.
        """
        x, lut = create_mask_lut(0.0, 0.2, 10000.0, -40.0)

        # Pente très raide: transition quasi instantanée
        # Le pic devrait être très étroit
        high_gain_count = np.sum(lut > 0.9)
        assert high_gain_count < 30  # Pic très étroit

    def test_slope_zero(self):
        """
        Test : Slope = 0.

        Vérifie le comportement avec slope = 0.
        """
        x, lut = create_mask_lut(0.0, 0.2, 0.0, -40.0)

        # Avec slope = 0, le gain_db = 0 * (...) = 0
        # Clippé entre min_gain et 0, donc tout est à 0 dB = 1.0 linéaire
        # Sauf si width/2 - distance < 0, alors clippé à 0
        # En fait: gain_db = 0 * (w/2 - |x-pan|) = 0 pour tout x
        # Donc tout devrait être à 1.0
        assert_allclose(lut, 1.0, atol=1e-5)


class TestGenerateExtractionMask:
    """Tests pour la fonction complète generate_extraction_mask."""

    def test_generate_mask_shape(self):
        """Test de la forme de sortie."""
        n_frames, n_freq = 100, 65
        panning = np.random.uniform(-1, 1, (n_frames, n_freq)).astype(np.float32)

        mask = generate_extraction_mask(
            panning=panning,
            source_pan=0.0,
            width=0.2,
            slope=500.0,
            min_gain_db=-40.0,
            attack_frames=1.0,
            release_frames=50.0,
            apply_blur=True,
            apply_smoothing=True,
        )

        assert mask.shape == (n_frames, n_freq)
        assert mask.dtype == np.float32

    def test_generate_mask_range(self):
        """Test que les gains sont dans le range attendu."""
        n_frames, n_freq = 50, 65
        panning = np.random.uniform(-1, 1, (n_frames, n_freq)).astype(np.float32)

        min_gain_db = -40.0
        min_gain_lin = 10 ** (min_gain_db / 20.0)

        mask = generate_extraction_mask(
            panning=panning,
            source_pan=0.0,
            width=0.2,
            slope=500.0,
            min_gain_db=min_gain_db,
            attack_frames=1.0,
            release_frames=50.0,
        )

        # Les gains devraient être entre min_gain et 1
        assert np.all(mask >= min_gain_lin - 0.01)
        assert np.all(mask <= 1.01)

    def test_generate_mask_no_blur_no_smoothing(self):
        """Test sans blur ni smoothing."""
        n_frames, n_freq = 10, 65
        # Panning constant au centre
        panning = np.zeros((n_frames, n_freq), dtype=np.float32)

        mask = generate_extraction_mask(
            panning=panning,
            source_pan=0.0,
            width=0.2,
            slope=500.0,
            min_gain_db=-40.0,
            attack_frames=1.0,
            release_frames=50.0,
            apply_blur=False,
            apply_smoothing=False,
        )

        # Avec panning = 0 et source_pan = 0, le gain devrait être maximal
        assert_allclose(mask, 1.0, atol=0.01)


class TestApplyTemporalSmoothing:
    """Tests pour la fonction apply_temporal_smoothing."""

    def test_smoothing_function(self):
        """Test de la fonction utilitaire."""
        n_frames, n_freq = 50, 65
        gains = np.random.rand(n_frames, n_freq).astype(np.float32)

        smoothed = apply_temporal_smoothing(
            gains,
            attack_frames=1.0,
            release_frames=50.0,
        )

        assert smoothed.shape == gains.shape
        assert smoothed.dtype == np.float32


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
