# Plan de Développement - Algorithme Upmix

## Phase 1 : RAFFINAGE DE LA SPÉCIFICATION

### 1.1 Répondre aux questions de clarification

- **Livrable** : Document `spec_detailed.md` avec toutes les réponses
- **Durée estimée** : 1-2 sessions
- **Actions** :
  - Répondre à toutes les questions du fichier `questions_clarification.md`
  - Documenter les choix techniques
  - Valider avec le concepteur de l'algorithme

### 1.2 Spécification technique détaillée

- **Livrable** : `spec_technique.md` avec :
  - Formules mathématiques précises
  - Algorithmes pseudo-code pour chaque étape
  - Structures de données
  - Interfaces de fonctions
  - Cas limites et gestion d'erreurs
- **Durée estimée** : 2-3 sessions
- **Sections** :
  1. Spécification des filtres IIR (formules cookbook exactes)
  2. Spécification STFT (paramètres, fenêtres, overlap-add)
  3. Spécification estimation panning (algorithme complet)
  4. Spécification extraction/masque (LUT, lissage)
  5. Spécification respatialisation (gains/délais)
  6. Structure JSON complète avec schéma de validation

### 1.3 Validation de la spécification

- **Livrable** : Spécification validée et signée
- **Actions** :
  - Revue de code de la spec
  - Tests de cohérence (vérifier que toutes les étapes s'enchaînent)
  - Vérification des unités (samples, ms, Hz, dB, etc.)

---

## Phase 2 : PLANS DE TESTS UNITAIRES

### 2.1 Architecture de tests

- **Structure** :

  ```
  tests/
  ├── unit/
  │   ├── test_biquad.py
  │   ├── test_crossover.py
  │   ├── test_lfe_creation.py
  │   ├── test_stft.py
  │   ├── test_panning_estimation.py
  │   ├── test_mask_generation.py
  │   ├── test_extraction.py
  │   ├── test_respatialization.py
  │   └── test_integration_steps.py
  ├── fixtures/
  │   ├── test_signals/
  │   └── test_params/
  └── integration/
      └── test_full_pipeline.py
  ```

### 2.2 Tests unitaires - Filtres Biquad (Module 1)

- **Fichier** : `test_biquad.py`
- **Tests** :
  - [ ] Calcul coefficients biquad (formule cookbook)
  - [ ] Réponse en fréquence : vérifier -6dB à F_xover1
  - [ ] Cascade de 2 biquads : réponse totale
  - [ ] Filtre passe-bas pour LFE : réponse à F_LFE
  - [ ] Stabilité numérique (pas de dépassement)
  - [ ] Phase linéaire (ou vérification de la phase)
- **Fixtures** : Signaux sinusoïdaux à différentes fréquences

### 2.3 Tests unitaires - Crossovers (Module 1)

- **Fichier** : `test_crossover.py`
- **Tests** :
  - [ ] Crossover stéréo : séparation L/R correcte
  - [ ] Somme à puissance constante : vérifier niveau RMS
  - [ ] Généralisation multicanal : tous canaux traités
  - [ ] Exclusion LFE : LFE non inclus dans somme
  - [ ] Conservation énergie : énergie totale préservée
- **Fixtures** : Signaux stéréo et multicanal de test

### 2.4 Tests unitaires - Création LFE (Module 2)

- **Fichier** : `test_lfe_creation.py`
- **Tests** :
  - [ ] Détection LFE existant : identification correcte
  - [ ] Création LFE depuis somme : niveau correct
  - [ ] Filtre LP LFE : atténuation au-delà de F_LFE
  - [ ] Cas multicanal : tous canaux inclus sauf LFE
- **Fixtures** : Signaux avec/sans LFE

### 2.5 Tests unitaires - STFT (Module 3)

- **Fichier** : `test_stft.py`
- **Tests** :
  - [ ] STFT forward : dimensions correctes
  - [ ] ISTFT inverse : reconstruction parfaite (signal identique)
  - [ ] Overlap-add : pas d'artefacts
  - [ ] Fenêtre duale : condition de reconstruction
  - [ ] Fenêtre Hann : valeurs correctes
  - [ ] Overlap 25% : hop_size correct
- **Fixtures** : Signaux de test (sinus, bruit blanc, musique)

### 2.6 Tests unitaires - Estimation Panning (Module 3)

- **Fichier** : `test_panning_estimation.py`
- **Tests** :
  - [ ] Calcul vecteur d'énergie : direction correcte
  - [ ] Normalisation angle : valeur entre -1 et 1
  - [ ] Stéréo : panning -1 (L) à +1 (R)
  - [ ] Multicanal : panning sur 360°
  - [ ] Cas limites : signal mono, signal silencieux
- **Fixtures** : STFT de signaux avec panning connu

### 2.7 Tests unitaires - Génération Masque (Module 3)

- **Fichier** : `test_mask_generation.py`
- **Tests** :
  - [ ] LUT masque : valeurs correctes selon pan/width/slope
  - [ ] Blur triangulaire : lissage fréquentiel correct
  - [ ] Rampsmooth : attack/release corrects
  - [ ] Min_gain : floor respecté
  - [ ] Cas limites : pan hors range, width très petit/grand
- **Fixtures** : Paramètres de test variés

### 2.8 Tests unitaires - Extraction (Module 3)

- **Fichier** : `test_extraction.py`
- **Tests** :
  - [ ] Sélection signal le plus proche : choix correct
  - [ ] Application gain lissé : multiplication correcte
  - [ ] ISTFT après extraction : signal temporel valide
  - [ ] Plusieurs sources : extraction indépendante
- **Fixtures** : STFT + masques de test

### 2.9 Tests unitaires - Respatialisation (Module 5)

- **Fichier** : `test_respatialization.py`
- **Tests** :
  - [ ] Calcul gains spatialisation : valeurs correctes
  - [ ] Application délais : timing correct
  - [ ] Somme sources : pas de saturation
  - [ ] Canal LFE : routage correct
  - [ ] Formats variés : stéréo→5.1, 5.1→7.1, etc.
- **Fixtures** : Sources + layouts de test

### 2.10 Tests d'intégration par étape

- **Fichier** : `test_integration_steps.py`
- **Tests** :
  - [ ] Étape 1→2 : crossovers + LFE
  - [ ] Étape 2→3 : LFE + upmix fréquentiel
  - [ ] Étape 3→4 : upmix + ajout LF_mono1
  - [ ] Étape 4→5 : sources + respatialisation
  - [ ] Pipeline complet : entrée → sortie

### 2.11 Tests d'intégration complets

- **Fichier** : `test_full_pipeline.py`
- **Tests** :
  - [ ] Stéréo → 5.1 : résultat cohérent
  - [ ] 5.1 → 7.1 : préservation qualité
  - [ ] Formats variés : tous layouts supportés
  - [ ] Fichiers réels : WAV in → WAV out
  - [ ] Performance : temps de traitement acceptable
  - [ ] Mémoire : pas de fuites mémoire

### 2.12 Tests de non-régression

- **Fichier** : `test_regression.py`
- **Tests** :
  - [ ] Signaux de référence : résultats identiques
  - [ ] Métriques audio : RMS, spectre, etc.
  - [ ] Comparaison avec implémentation de référence (si disponible)

---

## Phase 3 : IMPLÉMENTATION

### 3.1 Architecture du code

```
upmix_algorithm/
├── __init__.py
├── upmix_processor.py          # Classe principale
├── modules/
│   ├── __init__.py
│   ├── biquad_filter.py        # Filtres IIR biquad
│   ├── crossover.py             # Crossovers
│   ├── lfe_processor.py          # Création LFE
│   ├── stft_processor.py        # STFT/ISTFT
│   ├── panning_estimator.py     # Estimation panning
│   ├── mask_generator.py        # Génération masques
│   ├── extractor.py              # Extraction fréquentielle
│   └── respatializer.py          # Respatialisation
├── utils/
│   ├── __init__.py
│   ├── audio_io.py               # Lecture/écriture WAV
│   ├── json_loader.py            # Chargement paramètres
│   └── layout_utils.py           # Utilitaires layouts
└── config/
    └── schema.json               # Schéma validation JSON
```

### 3.2 Ordre d'implémentation (bottom-up)

#### Étape 3.2.1 : Utilitaires de base

- **Modules** : `audio_io.py`, `json_loader.py`, `layout_utils.py`
- **Tests** : Créer tests unitaires pour ces utilitaires (non prévus dans Phase 2, à ajouter si nécessaire)
- **Durée** : 1 session

#### Étape 3.2.2 : Filtres Biquad

- **Module** : `biquad_filter.py`
- **Fonctions** :
  - `compute_biquad_coeffs(freq, q, fs, filter_type)`
  - `BiquadFilter` (classe avec état)
- **Tests** : Implémenter `test_biquad.py` (~20 TODO)
  - Remplacer les `pass` par le code réel
  - Décommenter les imports
  - Vérifier tous les tests passent
- **Durée** : 2 sessions (module) + 1 session (tests)

#### Étape 3.2.3 : Crossovers

- **Module** : `crossover.py`
- **Fonctions** :
  - `apply_crossover(signal, freq, fs)`
  - `sum_power_constant(signals)`
- **Tests** : Implémenter `test_crossover.py` (~20 TODO)
- **Durée** : 2 sessions (module) + 1 session (tests)

#### Étape 3.2.4 : Création LFE

- **Module** : `lfe_processor.py`
- **Fonctions** :
  - `detect_lfe_channels(layout)`
  - `create_lfe(signals, layout, freq, fs)`
- **Tests** : Implémenter `test_lfe_creation.py` (~15 TODO)
- **Durée** : 1 session (module) + 0.5 session (tests)

#### Étape 3.2.5 : STFT

- **Module** : `stft_processor.py`
- **Fonctions** :
  - `STFTProcessor` (classe avec état)
  - `forward(signal)`
  - `inverse(stft)`
- **Tests** : Implémenter `test_stft.py` (~20 TODO)
- **Durée** : 2-3 sessions (module) + 1 session (tests)

#### Étape 3.2.6 : Estimation Panning

- **Module** : `panning_estimator.py`
- **Fonctions** :
  - `estimate_panning(stft_channels, input_layout)`
- **Tests** : Implémenter `test_panning_estimation.py` (~15 TODO)
- **Durée** : 2 sessions (module) + 1 session (tests)

#### Étape 3.2.7 : Génération Masque

- **Module** : `mask_generator.py`
- **Fonctions** :
  - `create_mask_lut(pan, width, slope, min_gain)`
  - `apply_freq_blur(mask, kernel_size)`
  - `apply_temporal_smoothing(mask, attack, release, fs)`
- **Tests** : Implémenter `test_mask_generation.py` (~20 TODO)
- **Durée** : 2 sessions (module) + 1 session (tests)

#### Étape 3.2.8 : Extraction

- **Module** : `extractor.py`
- **Fonctions** :
  - `extract_source(stft_channels, panning, mask_params, input_layout)`
- **Tests** : Implémenter `test_extraction.py` (~15 TODO)
- **Durée** : 2-3 sessions (module) + 1 session (tests)

#### Étape 3.2.9 : Respatialisation

- **Module** : `respatializer.py`
- **Fonctions** :
  - `compute_spatialization_gains(sources, output_layout, params)`
  - `apply_spatialization(sources, gains, delays, fs)`
- **Tests** : Implémenter `test_respatialization.py` (~20 TODO)
- **Durée** : 2-3 sessions (module) + 1 session (tests)

#### Étape 3.2.10 : Intégration - Classe principale

- **Module** : `upmix_processor.py`
- **Classe** : `UpmixProcessor`
- **Méthodes** :
  - `__init__(params_json, input_layout, output_layout)`
  - `process(input_wav_file, output_wav_file)`
  - Méthodes privées pour chaque étape
- **Tests** : Implémenter `test_integration_steps.py` (~15 TODO), `test_full_pipeline.py` (~15 TODO), `test_regression.py` (~10 TODO)
- **Durée** : 3-4 sessions (module) + 2-3 sessions (tests)

### 3.3 Implémentation des Tests

- **Stratégie** : Implémenter les tests **en parallèle** ou **immédiatement après** chaque module
- **Ordre** : Suivre l'ordre d'implémentation des modules (3.2.2 → 3.2.10)
- **TODO** : ~210 TODO à implémenter (voir `tests/TODO_TESTS.md` pour détails)
- **Actions pour chaque module** :
  1. Implémenter le module
  2. Décommenter les imports dans le fichier de test correspondant
  3. Remplacer les `pass` par le code réel des tests
  4. Exécuter les tests et corriger jusqu'à 100% de passage
  5. Vérifier la couverture de code

### 3.4 Bonnes pratiques

- **Code** :
  - Type hints (Python 3.8+)
  - Docstrings (Google style)
  - Logging pour debug
  - Gestion d'erreurs explicite
- **Tests** :
  - Couverture > 80%
  - Tests avant chaque commit
  - CI/CD si possible
  - Implémenter les tests au fur et à mesure (pas tous à la fin)
- **Documentation** :
  - README avec exemples
  - Docstrings complètes
  - Schémas si nécessaire

---

## Phase 4 : TESTS ET VALIDATION

### 4.1 Tests unitaires

- **Action** : Exécuter tous les tests unitaires
- **Objectif** : 100% de passage
- **Durée** : 1-2 sessions (corrections)

### 4.2 Tests d'intégration

- **Action** : Exécuter tests d'intégration
- **Objectif** : Pipeline complet fonctionnel
- **Durée** : 2-3 sessions (corrections)

### 4.3 Tests de performance

- **Métriques** :
  - Temps de traitement (secondes de traitement / secondes audio)
  - Utilisation mémoire
  - CPU usage
- **Objectif** : Performance acceptable (< 10x temps réel ?)

### 4.4 Tests audio

- **Tests subjectifs** :
  - Écoute de résultats
  - Comparaison avec référence
  - Détection artefacts
- **Tests objectifs** :
  - Métriques audio (SNR, PESQ si applicable)
  - Analyse spectrale
  - Vérification niveaux

### 4.5 Validation finale

- **Checklist** :
  - [ ] Tous les tests passent
  - [ ] Documentation complète
  - [ ] Exemples fonctionnels
  - [ ] Performance acceptable
  - [ ] Code review effectuée
  - [ ] Validation par le concepteur

---

## RÉSUMÉ DES LIVRABLES

1. **Spécification détaillée** (`spec_technique.md`)
2. **Code source** (modules Python)
3. **Tests unitaires** (pytest)
4. **Tests d'intégration** (pytest)
5. **Documentation** (README, docstrings)
6. **Exemples d'utilisation** (scripts, notebooks)
7. **Rapport de validation** (résultats tests, métriques)

---

## ESTIMATION TEMPORELLE TOTALE

- **Phase 1** (Raffinage spec) : 3-5 sessions
- **Phase 2** (Plans tests) : 2-3 sessions ✅ **TERMINÉE**
- **Phase 3** (Implémentation) : 20-30 sessions (modules) + 10-12 sessions (implémentation tests)
- **Phase 4** (Tests/Validation) : 5-8 sessions (corrections et validation finale)

**Total estimé** : 40-58 sessions de travail

*Note : Une session = 2-4 heures de travail concentré*

## TODO - Phase 2 Complétée

✅ **Structure de tests créée** : 11 fichiers de tests avec ~210 TODO
✅ **Fixtures pytest** : conftest.py avec fixtures réutilisables
✅ **Documentation** : README, PLAN_TESTS.md, TODO_TESTS.md

**Prochaines étapes** :

- Phase 3 : Implémenter les modules
- En parallèle : Implémenter les tests (remplacer les TODO)
- Voir `tests/TODO_TESTS.md` pour la liste complète des TODO
