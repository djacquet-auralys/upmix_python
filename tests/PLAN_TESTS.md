# Plan de Tests - Phase 2

## Structure Créée

```
tests/
├── README.md                    # Documentation des tests
├── conftest.py                  # Fixtures pytest partagées
├── unit/                        # Tests unitaires
│   ├── __init__.py
│   ├── test_biquad.py          # 6 classes, ~20 tests
│   ├── test_crossover.py       # 6 classes, ~20 tests
│   ├── test_lfe_creation.py    # 6 classes, ~15 tests
│   ├── test_stft.py            # 7 classes, ~20 tests
│   ├── test_panning_estimation.py  # 6 classes, ~15 tests
│   ├── test_mask_generation.py     # 5 classes, ~20 tests
│   ├── test_extraction.py          # 4 classes, ~15 tests
│   ├── test_respatialization.py    # 6 classes, ~20 tests
│   └── test_integration_steps.py   # 5 classes, ~15 tests
├── integration/                 # Tests d'intégration
│   ├── __init__.py
│   ├── test_full_pipeline.py   # 5 classes, ~15 tests
│   └── test_regression.py      # 3 classes, ~10 tests
└── fixtures/                    # Données de test
    ├── test_signals/           # Signaux audio (à créer)
    └── test_params/            # JSON de test (à créer)
```

## Résumé des Tests

### Tests Unitaires (~160 tests)

#### 1. test_biquad.py

- **6 classes de tests**
- Calcul coefficients (LPF, HPF)
- Réponse en fréquence (-6dB à coupure)
- Cascade ordre 4
- Stabilité numérique
- Phase

#### 2. test_crossover.py

- **6 classes de tests**
- Crossover stéréo
- Somme à puissance constante
- Généralisation multicanal
- Exclusion LFE
- Conservation énergie

#### 3. test_lfe_creation.py

- **6 classes de tests**
- Détection LFE
- Création depuis somme
- Filtre LP LFE
- Cas multicanal

#### 4. test_stft.py

- **7 classes de tests**
- STFT forward (dimensions)
- ISTFT inverse (reconstruction)
- Overlap-add
- Fenêtre duale (sqrt(hann))
- Overlap 25%

#### 5. test_panning_estimation.py

- **6 classes de tests**
- Vecteur d'énergie
- Normalisation angle
- Panning stéréo (-1 à +1)
- Panning multicanal (360°)
- Cas limites

#### 6. test_mask_generation.py

- **5 classes de tests**
- LUT masque (200 points)
- Blur triangulaire (3 bins)
- Rampsmooth (attack/release)
- Min_gain floor
- Cas limites

#### 7. test_extraction.py

- **4 classes de tests**
- Sélection signal
- Application gain
- ISTFT après extraction
- Plusieurs sources

#### 8. test_respatialization.py

- **6 classes de tests**
- Calcul gains (JSON ou TDAP)
- Application délais
- Somme sources
- Routage LFE
- Formats variés

#### 9. test_integration_steps.py

- **5 classes de tests**
- Étape 1→2 (crossovers + LFE)
- Étape 2→3 (LFE + upmix)
- Étape 3→4 (upmix + LF_mono1)
- Étape 4→5 (sources + respatialisation)
- Pipeline complet

### Tests d'Intégration (~25 tests)

#### 1. test_full_pipeline.py

- **5 classes de tests**
- Conversions de format
- Préservation qualité
- Fichiers WAV réels
- Performance
- Mémoire

#### 2. test_regression.py

- **3 classes de tests**
- Signaux de référence
- Métriques audio
- Reproductibilité

## Fixtures Disponibles

Définies dans `conftest.py` :

- `sample_rate` : 48000 Hz
- `duration_seconds` : 1.0 s
- `n_samples` : calculé automatiquement
- `stereo_signal` : signal stéréo sinusoïdal
- `multichannel_signal_5_1` : signal 5.1
- `white_noise_signal` : bruit blanc
- `sine_signal` : sinusoïde pure
- `default_params` : paramètres par défaut
- `upmix_params_json` : paramètres upmix de base
- `biquad_params` : paramètres biquad

## Prochaines Étapes

1. **Implémenter les modules** (Phase 3)
2. **Implémenter les tests** (remplacer les `pass` par le code réel)
3. **Créer les fixtures audio** (signaux de test dans `fixtures/test_signals/`)
4. **Créer les fixtures JSON** (paramètres de test dans `fixtures/test_params/`)
5. **Exécuter les tests** : `pytest tests/`

## Notes

- Tous les tests sont actuellement des **squelettes** avec `pass`
- Les imports sont commentés avec `# TODO:`
- Chaque test a une docstring expliquant ce qu'il teste
- Les tests suivent les conventions pytest
- Utilisation de `assert_allclose` pour les comparaisons numériques

## Commandes Utiles

```bash
# Exécuter tous les tests
pytest tests/

# Exécuter un fichier spécifique
pytest tests/unit/test_biquad.py

# Exécuter avec verbose
pytest tests/ -v

# Exécuter avec couverture
pytest tests/ --cov=upmix_algorithm --cov-report=html

# Exécuter seulement les tests unitaires
pytest tests/unit/

# Exécuter seulement les tests d'intégration
pytest tests/integration/
```
