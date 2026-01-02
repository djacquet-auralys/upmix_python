# Tests - Algorithme Upmix

## Structure

```
tests/
├── unit/                    # Tests unitaires par module
│   ├── test_biquad.py
│   ├── test_crossover.py
│   ├── test_lfe_creation.py
│   ├── test_stft.py
│   ├── test_panning_estimation.py
│   ├── test_mask_generation.py
│   ├── test_extraction.py
│   ├── test_respatialization.py
│   └── test_integration_steps.py
├── fixtures/                # Données de test
│   ├── test_signals/       # Signaux audio de test
│   └── test_params/        # Paramètres JSON de test
├── integration/            # Tests d'intégration
│   ├── test_full_pipeline.py
│   └── test_regression.py
└── conftest.py             # Configuration pytest partagée
```

## Exécution

```bash
# Tous les tests
pytest tests/

# Tests unitaires uniquement
pytest tests/unit/

# Tests d'intégration uniquement
pytest tests/integration/

# Un fichier spécifique
pytest tests/unit/test_biquad.py

# Avec couverture
pytest tests/ --cov=upmix_algorithm --cov-report=html
```

## Fixtures

Les fixtures sont définies dans `conftest.py` et peuvent être utilisées dans tous les tests via l'injection de dépendances pytest.
