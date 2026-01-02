# Points Clarifiés - Estimation de Panning

## Décisions pour l'implémentation de `re_model_light.py`

### ✅ 1. Simplification de `re.compute_re_model`

**Décision** : Créer un module `re_model_light.py` entièrement nouveau et simplifié.

- `seat` = uniquement position centrale (origine 0, 0)
- `speaker_axis` = position centrale (non utilisé)
- `directivity` = 100° mais sans impact (calcul dans l'axe → gain toujours 0 dB)
- `angle_att_dB` = toujours 0 dB (gain unitaire) → omis
- `st_levels` = toujours unitaire (0 dB) → calcul supprimé
- `spat_delays` = toujours 0 → omis

**Implémenté** : Nouveau module `upmix_algorithm/modules/re_model_light.py`

### ✅ 2. Coordonnées des canaux d'entrée

**Décision** :

- Utiliser `multichannel_layouts.get_spk_coordinates(layout, radius=1.0)`
- Le rayon = 1.0 (vecteurs unitaires, seule la direction compte)
- Passer le **nom du layout** (string) : `"stereo"`, `"5.1"`, etc.

### ✅ 3. Détermination Stéréo vs Multicanal

**Décision** : Par le nom du layout

- Si `layout == "stereo"` → normalisation par 60° (±30°)
- Sinon → normalisation par 360° (cercle complet)

### ✅ 4. Format d'entrée/sortie

**API implémentée** :

```python
def estimate_panning(
    stft_magnitudes: np.ndarray,  # (n_frames, n_freq, n_channels) linéaire
    layout: str,                   # "stereo", "5.1", etc.
    epsilon: float = 1e-12
) -> np.ndarray:  # (n_frames, n_freq) avec valeurs [-1, 1]
```

### ✅ 5. Conversion Gains → dB

**Décision** : **Pas de conversion dB** pour l'efficacité.

- Le calcul se fait entièrement en linéaire
- Formule : `RE = sum(g² × v) / sum(g²)`

### ✅ 6. Calcul de l'angle final

**Formule implémentée** :

1. Calcul angle : `angle_rad = atan2(x, y)` (0° = avant +y)
2. Conversion en degrés : `angle_deg = rad2deg(angle_rad)`
3. Normalisation : `pan = angle_deg / (range/2)` où range = 60° ou 360°
4. Clipping : `pan = clip(pan, -1, 1)`

### ✅ 7. Gestion des canaux LFE

**Décision** : Les canaux LFE sont **automatiquement exclus** du calcul.

- Identification par label contenant "LFE"
- Index LFE retourné par `get_speaker_unit_vectors()`

### ✅ 8. Cas limites

**Décisions** :

- Signal silencieux (gains nuls) : `epsilon = 1e-12` évite division par zéro
- Vecteur d'énergie (0, 0) : impossible grâce à epsilon
- Mono (1 canal) : non supporté (minimum 2 canaux requis)

---

## Module implémenté

**Fichier** : `upmix_algorithm/modules/re_model_light.py`

**Fonctions principales** :

- `estimate_panning(stft_magnitudes, layout)` → panning [-1, 1]
- `compute_energy_vector(gains, unit_vectors)` → vecteur RE (x, y)
- `energy_vector_to_angle(re, normalize_range)` → angle normalisé
- `get_speaker_unit_vectors(layout)` → vecteurs unitaires des HP
- `estimate_source_width(stft_magnitudes, layout)` → largeur en degrés
- `get_layout_info(layout)` → info sur le layout

**Tests** : `tests/unit/test_re_model_light.py` (32 tests, tous passent)
