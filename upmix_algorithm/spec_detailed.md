# Spécification Détaillée - Algorithme Upmix

## Réponses aux Questions de Clarification

### 1. CROSSOVERS (Étape 1)

#### 1.1 Filtres IIR Biquad

- **Q1.1.1** ✅ **RÉPONSE** : Formules du [W3C Audio EQ Cookbook](https://www.w3.org/TR/audio-eq-cookbook/#formulae)
- **Q1.1.2** ✅ **RÉPONSE** : Types HPF (High Pass Filter) et LPF (Low Pass Filter) tels que formulés dans le document
- **Q1.1.3** ✅ **RÉPONSE** : Filtre à -6 dB à la fréquence de coupure, obtenu avec un ordre 4 en superposant 2 HP ou 2 LP
- **Q1.1.4** ✅ **RÉPONSE** : Oui, réglage identique pour les 2 biquads en cascade

#### 1.2 Somme à puissance constante

- **Q1.2.1** ✅ **RÉPONSE** : Option A : `LF_mono1 = sqrt(sum(L_lowfreq² + R_lowfreq² + ...))`
- **Q1.2.2** ✅ **RÉPONSE** : Non, le facteur 0.707 est uniquement valable pour de la stéréo

#### 1.3 Généralisation multicanal

- **Q1.3.1** ✅ **RÉPONSE** : Exclure les LFE
- **Q1.3.2** ✅ **RÉPONSE** : Oui, on exclut le LFE de la somme

### 2. CRÉATION CANAL LFE (Étape 2)

#### 2.1 Détection LFE existant

- **Q2.1.1** ✅ **RÉPONSE** : Par le label dans le fichier multichannel_layouts.py
- **Q2.1.2** ✅ **RÉPONSE** : On somme en mono. **TODO** : Réfléchir à ce qu'il faut faire par la suite

#### 2.2 Création LFE depuis somme mono

- **Q2.2.1** ✅ **RÉPONSE** : Même réponse que Q1.2.1 (Option A)
- **Q2.2.2** ✅ **RÉPONSE** : Tous sauf LFE. **TODO** : Réfléchir à ce qu'il faut faire par la suite
- **Q2.2.3** ✅ **RÉPONSE** : Oui, 2 biquads en cascade, Q = 0.707

### 3. UPMIX FRÉQUENTIEL (Étape 3)

#### 3.1 Calcul nombre de sources

- **Q3.1.1** ✅ **RÉPONSE** :
  - `nb_spk` = nombre de HP de destination (hors LFE)
  - `max_sources` = 11 si absent du JSON
- **Q3.1.2** ✅ **RÉPONSE** : Oui, on garde uniquement celles calculées

#### 3.2 Paramètres d'extraction

- **Q3.2.1** ✅ **RÉPONSE** :
  - `gains[i]` = gain pour le HP i de destination
  - `delays[i]` = délai en ms
- **Q3.2.2** ✅ **RÉPONSE** : Exact, si mute=1, la source est complètement ignorée
- **Q3.2.3** ✅ **RÉPONSE** : À priori en frames STFT

#### 3.3 STFT

- **Q3.3.1** ✅ **RÉPONSE** :
  - `nwin` = taille de la fenêtre = 128 (par défaut)
  - `nfreq` = nombre de bins fréquentiels = 128/2 + 1 = 65
  - Paramètres du script (réglables)
- **Q3.3.2** ✅ **RÉPONSE** : hop_size = nwin * 0.25 = 32 samples
- **Q3.3.3** ✅ **RÉPONSE** : À vérifier, mais à priori on prend sqrt(hann) pour avoir la même fenêtre à l'entrée et à la sortie

#### 3.4 Estimation de panning

- **Q3.4.1** ✅ **RÉPONSE** : On utilise `re.compute_re_model` pour obtenir l'estimation de panning comme précisé dans la spec. Sauf que les gains sont remplacés par les modules des STFT à chaque fréquence
- **Q3.4.2** ✅ **RÉPONSE** : Oui, on considère les délais nuls
- **Q3.4.3** ✅ **RÉPONSE** : Oui, pour chaque canal d'entrée, on prend |STFT| comme "gain"
- **Q3.4.4** ✅ **RÉPONSE** : x et y sont les coordonnées du vecteur d'énergie obtenu pour chaque fréquence. On doit connaître les coordonnées des canaux d'entrée.
- **Q3.4.5** ✅ **RÉPONSE** : 60° ou 360° uniquement (pas de calcul depuis les azimuts réels)

#### 3.5 Extraction - Masque

- **Q3.5.1** ✅ **RÉPONSE** : Correct, floor = min_gain
- **Q3.5.2** ✅ **RÉPONSE** : 200 points, interpolation linéaire
- **Q3.5.3** ✅ **RÉPONSE** : 3 bins, linéaire décroissant
- **Q3.5.4** ✅ **RÉPONSE** : Voir code de référence fourni (issu de gen~ dans RNBO). Algorithme de rampsmooth avec freeze si power < 1e-6, doublage du release pour bin 0 et Nyquist
- **Q3.5.5** ✅ **RÉPONSE** : Oui, on détermine les angles depuis le format d'entrée
- **Q3.5.6** ✅ **RÉPONSE** : Oui, multiplication complexe

#### 3.6 STFT inverse et overlap-add

- **Q3.6.1** ✅ **RÉPONSE** : sqrt(hann) pour fenêtre duale
- **Q3.6.2** ✅ **RÉPONSE** : Pas de modulation ou variation d'amplitude significative, overlap-add avec normalisation

### 4. AJOUT LF_MONO1 (Étape 4)

#### 4.1 Délai de latence

- **Q4.1.1** ✅ **RÉPONSE** : C'est la valeur utilisée dans MaxMSP. Il faut que le signal généré par le traitement fréquentiel et le signal original soient alignés temporellement, en phase
- **Q4.1.2** ✅ **RÉPONSE** : En samples

#### 4.2 Application du gain

- **Q4.2.1** ✅ **RÉPONSE** : En dB
- **Q4.2.2** ✅ **RÉPONSE** : Exact, le signal LF_mono1 retardé est sommé à CHAQUE signal extrait (chaque source) avec son propre `LF_gain[i]`

### 5. RESPATIALISATION (Étape 5)

#### 5.1 Placement des sources

- **Q5.1.1** ✅ **RÉPONSE** :
  - Les coordonnées des spk de destination sont données par le format de destination
  - audience_bary = origine du système
  - panorama_center = speaker central ou la position équivalente à un panning = 0
  - Paramètres dans le JSON (un JSON plus clair sera fourni)
- **Q5.1.2** ⚠️ **À RÉFLÉCHIR** : À voir, il faut réfléchir là-dessus. Soit on fait des fichiers JSON par configuration d'arrivée ou c'est déjà défini (le mieux), soit on recalcule à chaque fois

#### 5.2 Calcul gains de spatialisation

- **Q5.2.1** ✅ **RÉPONSE** :
  - Si déjà calculés, on les applique directement
  - Sinon, on utilise TDAP
- **Q5.2.2** ✅ **RÉPONSE** : Délais en ms

#### 5.3 Application aux canaux de sortie

- **Q5.3.1** ✅ **RÉPONSE** : Oui, on somme toutes les sources avec leurs gains/délais respectifs
- **Q5.3.2** ✅ **RÉPONSE** : Délais entiers en samples
- **Q5.3.3** ✅ **RÉPONSE** : Oui, le canal LFE de sortie reçoit directement le canal LFE créé à l'étape 2

### 6. FORMATS ET PARAMÈTRES GLOBAUX

#### 6.1 Formats d'entrée/sortie

- **Q6.1.1** ✅ **RÉPONSE** : Oui, ce sera un dropdown dans une UI
- **Q6.1.2** ✅ **RÉPONSE** : Idem
- **Q6.1.3** ✅ **RÉPONSE** : Ordre standard donné dans multichannel_layouts.py

#### 6.2 Structure JSON

- **Q6.2.1** ✅ **RÉPONSE** : Oui, on peut partir là-dessus (structure proposée validée)
- **Q6.2.2** ⚠️ **TODO** : À voir, TODO

#### 6.3 Traitement des erreurs

- **Q6.3.1** ✅ **RÉPONSE** : On détecte le nombre de canaux et on propose les formats possibles, on demande confirmation
- **Q6.3.2** ⚠️ **TODO** : À voir, TODO
- **Q6.3.3** ✅ **RÉPONSE** : On retourne une erreur et on arrête

### 7. DÉTAILS TECHNIQUES

#### 7.1 Bibliothèques

- **Q7.1.1** ✅ **RÉPONSE** : scipy pour tout

#### 7.2 Précision numérique

- **Q7.2.1** ✅ **RÉPONSE** : float32
- **Q7.2.2** ✅ **RÉPONSE** : Oui, on garde le niveau RMS

#### 7.3 Performance

- **Q7.3.1** ✅ **RÉPONSE** : Non, pas de contrainte, c'est de la conversion, donc offline
- **Q7.3.2** ✅ **RÉPONSE** : Oui, traitement par blocs internes autorisé

---

## Points à Clarifier / TODOs

### TODOs Identifiés

1. **TODO LFE multiple (Q2.1.2)** : Si plusieurs canaux LFE existent, on somme en mono. À réfléchir sur ce qu'il faut faire par la suite.

2. **TODO Canaux pour création LFE (Q2.2.2)** : Tous sauf LFE. À réfléchir sur ce qu'il faut faire par la suite.

3. **TODO Placement sources (Q5.1.2)** : À voir, il faut réfléchir là-dessus. Soit on fait des fichiers JSON par configuration d'arrivée ou c'est déjà défini (le mieux), soit on recalcule à chaque fois.

4. **TODO Paramètres optionnels JSON (Q6.2.2)** : À voir, TODO

5. **TODO JSON avec plus de sources (Q6.3.2)** : À voir, TODO

### Points à Vérifier

1. **Fenêtre STFT (Q3.3.3)** : À vérifier, mais à priori sqrt(hann) pour avoir la même fenêtre à l'entrée et à la sortie.

2. **Unités attack/release (Q3.2.3)** : À priori en frames STFT, mais à confirmer lors de l'implémentation.

3. **JSON plus clair (Q5.1.1)** : Un JSON plus clair sera fourni avec les paramètres de respatialisation.

---

## Détails Techniques Complémentaires

### Code de Référence - Lissage Temporel

Le code de référence fourni pour le lissage temporel (Q3.5.4) implémente :

- Ramp-up et ramp-down avec frames configurables
- Freeze si power < 1e-6
- Doublage du release pour bin 0 et Nyquist
- Blur triangulaire ±1 bin (hors DC & Nyquist)
- Traitement des paires de bins conjugués

### Formules Biquad (W3C Audio EQ Cookbook)

Pour LPF (Low Pass Filter) :

```
ω₀ = 2π × f₀ / Fₛ
α = sin(ω₀) / (2 × Q)
b₀ = (1 - cos(ω₀)) / 2
b₁ = 1 - cos(ω₀)
b₂ = (1 - cos(ω₀)) / 2
a₀ = 1 + α
a₁ = -2cos(ω₀)
a₂ = 1 - α
```

Pour HPF (High Pass Filter) :

```
b₀ = (1 + cos(ω₀)) / 2
b₁ = -(1 + cos(ω₀))
b₂ = (1 + cos(ω₀)) / 2
a₀ = 1 + α
a₁ = -2cos(ω₀)
a₂ = 1 - α
```

### Structure JSON Attendue

```json
{
  "input_layout": "stereo",
  "output_layout": "7.1",
  "F_xover1": 150.0,
  "F_LFE": 120.0,
  "max_sources": 11,
  "nfft": 128,
  "overlap": 0.25,
  "audience_bary": [0.0, 0.0],
  "panorama_center": [0.0, 0.0],
  "panorama_width": 200.0,
  "upmix_params": {
    "width": 0.18,
    "slope": 500.0,
    "min_gain": -40.0,
    "attack": 1.0,
    "pan1": 0.91,
    "gains1": [...],
    "delays1": [...],
    "release1": 186.36,
    "mute1": 0,
    "LF_gain1": 1.0,
    ...
  }
}
```


