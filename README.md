# Auralys Upmix Algorithm

Algorithme d'upmix audio pour conversion de signaux stéréo/multicanal vers des configurations surround (5.1, 7.1, etc.).

## Description

Ce projet implémente un algorithme d'upmix basé sur l'analyse fréquentielle (STFT), l'estimation de panning, l'extraction de sources et la respatialisation vers des layouts multicanal cibles.

## Structure du Projet

```
auralys_upmix/
├── upmix_algorithm/          # Code principal de l'algorithme
│   ├── upmix_processor.py    # Processeur principal (classe UpmixProcessor)
│   ├── modules/              # Modules de traitement audio
│   │   ├── biquad_filter.py  # Filtres IIR Biquad (LPF, HPF, PK, Shelves)
│   │   ├── crossover.py      # Crossovers et somme à puissance constante
│   │   ├── lfe_processor.py  # Traitement du canal LFE
│   │   ├── stft_processor.py # STFT/ISTFT avec fenêtre duale
│   │   ├── re_model_light.py # Estimation de panning (vecteur d'énergie)
│   │   ├── mask_generator.py # Génération et lissage des masques d'extraction
│   │   ├── extractor.py      # Extraction de sources fréquentielles
│   │   └── respatializer.py  # Respatialisation vers layout de sortie
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

### Upmix complet (stéréo → 5.1)

```python
from upmix_algorithm import UpmixProcessor, create_default_params
import numpy as np

# Créer les paramètres par défaut
params = create_default_params(
    input_layout="stereo",
    output_layout="5.1",
    n_sources=5
)

# Initialiser le processeur
processor = UpmixProcessor(
    params=params,
    input_layout="stereo",
    output_layout="5.1",
    sample_rate=48000.0
)

# Traiter un signal stéréo (n_samples, 2)
input_signal = np.random.randn(48000, 2).astype(np.float32) * 0.1
output_signal = processor.process(input_signal)  # (n_samples, 6)

# Ou traiter un fichier WAV
processor.process_file("input_stereo.wav", "output_51.wav")
```

### Utilisation des modules individuels

```python
from upmix_algorithm.modules import STFTProcessor, Crossover, estimate_panning
import numpy as np

# STFT
stft_proc = STFTProcessor(nfft=128, overlap=0.25)
stft = stft_proc.forward(audio_signal)
reconstructed = stft_proc.inverse(stft)

# Estimation de panning
stft_magnitudes = np.abs(stft)  # (n_frames, n_freq, n_channels)
panning = estimate_panning(stft_magnitudes, layout="stereo")
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
- ✅ `extractor.py` - Extraction de sources fréquentielles
- ✅ `respatializer.py` - Respatialisation vers layout de sortie
- ✅ `upmix_processor.py` - Processeur principal d'intégration

### Tous les modules sont implémentés ! 🎉

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
