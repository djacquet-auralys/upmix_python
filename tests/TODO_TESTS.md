# TODO - Implémentation des Tests

## Résumé

**Total de TODO identifiés** : ~210 TODO dans les fichiers de test

Tous les tests sont actuellement des **squelettes** avec `pass` et des commentaires `# TODO:`.
Les tests doivent être implémentés **après** l'implémentation des modules correspondants (Phase 3).

## TODO par Module

### 1. test_biquad.py (~20 TODO)

- Importer les fonctions une fois implémentées
- Implémenter tous les tests (coefficients, réponse fréquentielle, cascade, stabilité, phase)

### 2. test_crossover.py (~20 TODO)

- Importer les fonctions une fois implémentées
- Implémenter tous les tests (crossover stéréo, somme puissance constante, multicanal, exclusion LFE, conservation énergie)

### 3. test_lfe_creation.py (~15 TODO)

- Importer les fonctions une fois implémentées
- Implémenter tous les tests (détection LFE, création, filtre LP, cas multicanal)

### 4. test_stft.py (~20 TODO)

- Importer les fonctions une fois implémentées
- Implémenter tous les tests (STFT forward, ISTFT inverse, overlap-add, fenêtre duale)

### 5. test_panning_estimation.py (~15 TODO)

- Importer les fonctions une fois implémentées
- Implémenter tous les tests (vecteur d'énergie, normalisation, panning stéréo/multicanal)

### 6. test_mask_generation.py (~20 TODO)

- Importer les fonctions une fois implémentées
- Implémenter tous les tests (LUT masque, blur triangulaire, rampsmooth, min_gain)

### 7. test_extraction.py (~15 TODO)

- Importer les fonctions une fois implémentées
- Implémenter tous les tests (sélection signal, application gain, ISTFT, plusieurs sources)

### 8. test_respatialization.py (~20 TODO)

- Importer les fonctions une fois implémentées
- Implémenter tous les tests (gains spatialisation, délais, somme sources, routage LFE)

### 9. test_integration_steps.py (~15 TODO)

- Importer les modules une fois implémentés
- Implémenter tous les tests (intégration par étape, pipeline complet)

### 10. test_full_pipeline.py (~15 TODO)

- Importer le processeur principal une fois implémenté
- Implémenter tous les tests (conversions format, qualité, WAV, performance, mémoire)

### 11. test_regression.py (~10 TODO)

- Importer le processeur une fois implémenté
- Implémenter tous les tests (signaux référence, métriques audio, reproductibilité)

## Bibliothèques Importées

### Utilisées ✓

- `numpy` : utilisé partout
- `pytest` : utilisé partout
- `numpy.testing.assert_allclose` : utilisé dans plusieurs tests
- `time` : utilisé dans test_full_pipeline.py
- `os` : utilisé dans test_full_pipeline.py
- `psutil` : utilisé dans test_full_pipeline.py (tests mémoire)

### Importées mais Non Utilisées Actuellement

#### 1. `assert_array_less` (test_biquad.py)

- **Statut** : Importé mais pas utilisé
- **Utilisation future** : Oui, sera utilisé pour vérifier que valeurs < seuil
- **Action** : Garder l'import

#### 2. `json` (conftest.py)

- **Statut** : Importé mais pas utilisé
- **Utilisation future** : Oui, sera utilisé pour charger les fixtures JSON de test
- **Action** : Garder l'import

#### 3. `Path` (conftest.py, test_full_pipeline.py, test_regression.py)

- **Statut** : Importé mais pas utilisé actuellement
- **Utilisation future** : Oui, sera utilisé pour :
  - Charger les fixtures audio (fichiers WAV de test)
  - Créer des fichiers temporaires pour les tests WAV
  - Gérer les chemins de fichiers
- **Action** : Garder l'import

## Ordre d'Implémentation des Tests

Les tests doivent être implémentés **dans le même ordre** que les modules (Phase 3) :

1. **Après biquad_filter.py** → Implémenter test_biquad.py
2. **Après crossover.py** → Implémenter test_crossover.py
3. **Après lfe_processor.py** → Implémenter test_lfe_creation.py
4. **Après stft_processor.py** → Implémenter test_stft.py
5. **Après panning_estimator.py** → Implémenter test_panning_estimation.py
6. **Après mask_generator.py** → Implémenter test_mask_generation.py
7. **Après extractor.py** → Implémenter test_extraction.py
8. **Après respatializer.py** → Implémenter test_respatialization.py
9. **Après intégration des modules** → Implémenter test_integration_steps.py
10. **Après upmix_processor.py** → Implémenter test_full_pipeline.py et test_regression.py

## Notes

- Tous les imports sont prêts et corrects
- Les bibliothèques non utilisées actuellement seront utilisées lors de l'implémentation
- Les fixtures dans `conftest.py` sont prêtes à être utilisées
- Les TODO sont détaillés dans chaque fichier de test avec des docstrings explicatives
