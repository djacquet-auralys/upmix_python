# Auralys Upmix Algorithm

Algorithme d'upmix audio pour conversion de signaux stéréo/multicanal vers des configurations surround (5.1, 7.1, etc.).

## Description

Ce projet implémente un algorithme d'upmix basé sur l'analyse fréquentielle (STFT), l'estimation de panning, l'extraction de sources et la respatialisation vers des layouts multicanal cibles.

## Structure du Projet

```
auralys_upmix/
├── upmix_algorithm/          # Code principal de l'algorithme
│   ├── modules/              # Modules de traitement audio
│   │   ├── biquad_filter.py  # Filtres IIR Biquad (LPF, HPF, PK, Shelves)
│   │   ├── crossover.py      # Crossovers et somme à puissance constante
│   │   ├── lfe_processor.py  # Traitement du canal LFE
│   │   ├── stft_processor.py # STFT/ISTFT avec fenêtre duale
│   │   ├── re_model_light.py # Estimation de panning (vecteur d'énergie)
│   │   └── mask_generator.py # Génération et lissage des masques d'extraction
│   ├── utils/                # Utilitaires
│   └── spec_detailed.md      # Spécification détaillée
├── tests/                    # Tests unitaires et d'intégration
│   ├── unit/                 # Tests unitaires par module
│   └── integration/          # Tests d'intégration
└── requirements.txt          # Dépendances Python
```

## Installation

### Prérequis

- Python 3.8+
- pip

### Installation

1. Cloner le dépôt :

```bash
git clone https://github.com/djacquet-auralys/upmix_python.git
cd upmix_python
```

2. Créer un environnement virtuel :

```bash
python -m venv .venv
```

3. Activer l'environnement virtuel :

- Windows (PowerShell) :

```powershell
.venv\Scripts\Activate.ps1
```

- Linux/Mac :

```bash
source .venv/bin/activate
```

4. Installer les dépendances :

```bash
pip install -r requirements.txt
```

## Utilisation

```python
from upmix_algorithm.modules import STFTProcessor, Crossover, LFEProcessor, estimate_panning

# Exemple d'utilisation STFT
processor = STFTProcessor(nfft=128, overlap=0.25)
stft = processor.forward(audio_signal)
reconstructed = processor.inverse(stft)

# Estimation de panning à partir des magnitudes STFT
import numpy as np
stft_magnitudes = np.abs(stft)  # (n_frames, n_freq, n_channels)
panning = estimate_panning(stft_magnitudes, layout="stereo")  # (n_frames, n_freq) dans [-1, 1]
```

## Tests

Exécuter tous les tests :

```bash
pytest tests/ -v
```

Avec couverture de code :

```bash
pytest tests/ --cov=upmix_algorithm --cov-report=html
```

## Développement

Voir `upmix_algorithm/plan_developpement.md` pour le plan de développement détaillé.

### Modules implémentés

- ✅ `biquad_filter.py` - Filtres IIR Biquad (LPF, HPF, PK, Low/High Shelf)
- ✅ `crossover.py` - Crossovers et somme à puissance constante
- ✅ `lfe_processor.py` - Traitement du canal LFE
- ✅ `stft_processor.py` - STFT/ISTFT avec fenêtre duale sqrt(hann)
- ✅ `re_model_light.py` - Estimation de panning (vecteur d'énergie RE)
- ✅ `mask_generator.py` - Génération et lissage des masques d'extraction

### Modules en développement

- ⏳ `extractor.py` - Extraction de sources fréquentielles
- ⏳ `extractor.py` - Extraction de sources
- ⏳ `respatializer.py` - Respatialisation
- ⏳ `upmix_processor.py` - Processeur principal d'intégration

## Spécifications

- **Format d'entrée** : WAV (stéréo ou multicanal)
- **Format de sortie** : WAV (5.1, 7.1, etc.)
- **Fréquence d'échantillonnage** : 48 kHz (configurable)
- **Précision** : float32
- **STFT** : nfft=128, overlap=25%, fenêtre sqrt(hann)
- **Filtres** : IIR Biquad, ordre 4 (2 biquads en cascade)

## Licence

[À définir]

## Auteurs

[À compléter]
