# Questions de Clarification - Spécification Algorithme Upmix

**STATUT** : ✅ Toutes les questions ont été répondues. Voir `spec_detailed.md` pour les réponses complètes.

## 1. CROSSOVERS (Étape 1)

### 1.1 Filtres IIR Biquad

- **Q1.1.1** ✅ **RÉPONDU** : [W3C Audio EQ Cookbook](https://www.w3.org/TR/audio-eq-cookbook/#formulae)
- **Q1.1.2** ✅ **RÉPONDU** : Types HPF et LPF tels que formulés dans le document
- **Q1.1.3** ✅ **RÉPONDU** : Filtre à -6 dB à la fréquence de coupure, obtenu avec ordre 4 (2 HP ou 2 LP superposés)
- **Q1.1.4** ✅ **RÉPONDU** : Oui, réglage identique pour les 2 biquads

### 1.2 Somme à puissance constante

- **Q1.2.1** ✅ **RÉPONDU** : Option A : `LF_mono1 = sqrt(sum(L_lowfreq² + R_lowfreq² + ...))`
- **Q1.2.2** ✅ **RÉPONDU** : Non, le facteur 0.707 est uniquement valable pour stéréo

### 1.3 Généralisation multicanal

- **Q1.3.1** ✅ **RÉPONDU** : Exclure les LFE
- **Q1.3.2** ✅ **RÉPONDU** : Oui, on exclut le LFE de la somme

## 2. CRÉATION CANAL LFE (Étape 2)

### 2.1 Détection LFE existant

- **Q2.1.1** ✅ **RÉPONDU** : Par le label dans multichannel_layouts.py
- **Q2.1.2** ✅ **RÉPONDU** : On somme en mono. ⚠️ **TODO** : Réfléchir à ce qu'il faut faire par la suite

### 2.2 Création LFE depuis somme mono

- **Q2.2.1** ✅ **RÉPONDU** : Même réponse que Q1.2.1 (Option A)
- **Q2.2.2** ✅ **RÉPONDU** : Tous sauf LFE. ⚠️ **TODO** : Réfléchir à ce qu'il faut faire par la suite
- **Q2.2.3** ✅ **RÉPONDU** : Oui, 2 biquads en cascade, Q = 0.707

## 3. UPMIX FRÉQUENTIEL (Étape 3)

### 3.1 Calcul nombre de sources

- **Q3.1.1** ✅ **RÉPONDU** :
  - `nb_spk` = nombre de HP de destination (hors LFE)
  - `max_sources` = 11 si absent du JSON
- **Q3.1.2** ✅ **RÉPONDU** : Oui, on garde uniquement celles calculées

### 3.2 Paramètres d'extraction

- **Q3.2.1** ✅ **RÉPONDU** :
  - `gains[i]` = gain pour le HP i de destination
  - `delays[i]` = délai en ms
- **Q3.2.2** ✅ **RÉPONDU** : Exact, si mute=1, la source est complètement ignorée
- **Q3.2.3** ✅ **RÉPONDU** : À priori en frames STFT

### 3.3 STFT

- **Q3.3.1** ✅ **RÉPONDU** :
  - `nwin` = taille de la fenêtre = 128 (par défaut)
  - `nfreq` = nombre de bins fréquentiels = 128/2 + 1 = 65
  - Paramètres du script (réglables)
- **Q3.3.2** ✅ **RÉPONDU** : hop_size = nwin * 0.25 = 32 samples
- **Q3.3.3** ⚠️ **À VÉRIFIER** : À vérifier, mais à priori sqrt(hann) pour avoir la même fenêtre à l'entrée et à la sortie

### 3.4 Estimation de panning

- **Q3.4.1** ✅ **RÉPONDU** : On utilise `re.compute_re_model` pour obtenir l'estimation de panning. Les gains sont remplacés par les modules des STFT à chaque fréquence
- **Q3.4.2** ✅ **RÉPONDU** : Oui, on considère les délais nuls
- **Q3.4.3** ✅ **RÉPONDU** : Oui, pour chaque canal d'entrée, on prend |STFT| comme "gain"
- **Q3.4.4** ✅ **RÉPONDU** : x et y sont les coordonnées du vecteur d'énergie obtenu pour chaque fréquence. On doit connaître les coordonnées des canaux d'entrée
- **Q3.4.5** ✅ **RÉPONDU** : 60° ou 360° uniquement (pas de calcul depuis azimuts réels)

### 3.5 Extraction - Masque

- **Q3.5.1** ✅ **RÉPONDU** : Correct, floor = min_gain
- **Q3.5.2** ✅ **RÉPONDU** : 200 points, interpolation linéaire
- **Q3.5.3** ✅ **RÉPONDU** : 3 bins, linéaire décroissant
- **Q3.5.4** ✅ **RÉPONDU** : Voir code de référence fourni (issu de gen~ dans RNBO). Algorithme rampsmooth avec freeze si power < 1e-6, doublage release pour bin 0 et Nyquist
- **Q3.5.5** ✅ **RÉPONDU** : Oui, on détermine les angles depuis le format d'entrée
- **Q3.5.6** ✅ **RÉPONDU** : Oui, multiplication complexe

### 3.6 STFT inverse et overlap-add

- **Q3.6.1** ✅ **RÉPONDU** : sqrt(hann) pour fenêtre duale
- **Q3.6.2** ✅ **RÉPONDU** : Pas de modulation ou variation d'amplitude significative, overlap-add avec normalisation

## 4. AJOUT LF_MONO1 (Étape 4)

### 4.1 Délai de latence

- **Q4.1.1** ✅ **RÉPONDU** : C'est la valeur utilisée dans MaxMSP. Il faut que le signal généré par le traitement fréquentiel et le signal original soient alignés temporellement, en phase
- **Q4.1.2** ✅ **RÉPONDU** : En samples

### 4.2 Application du gain

- **Q4.2.1** ✅ **RÉPONDU** : En dB
- **Q4.2.2** ✅ **RÉPONDU** : Exact, le signal LF_mono1 retardé est sommé à CHAQUE signal extrait avec son propre LF_gain[i]

## 5. RESPATIALISATION (Étape 5)

### 5.1 Placement des sources

- **Q5.1.1** ✅ **RÉPONDU** :
  - Coordonnées HP destination : données par le format de destination
  - audience_bary : origine du système
  - panorama_center : speaker central ou position équivalente à panning = 0
  - Paramètres dans le JSON (un JSON plus clair sera fourni)
- **Q5.1.2** ⚠️ **TODO** : À voir, il faut réfléchir là-dessus. Soit fichiers JSON par configuration, soit déjà défini (le mieux), soit recalcul à chaque fois

### 5.2 Calcul gains de spatialisation

- **Q5.2.1** ✅ **RÉPONDU** :
  - Si déjà calculés, on les applique directement
  - Sinon, on utilise TDAP
- **Q5.2.2** ✅ **RÉPONDU** : Délais en ms

### 5.3 Application aux canaux de sortie

- **Q5.3.1** ✅ **RÉPONDU** : Oui, on somme toutes les sources avec leurs gains/délais respectifs
- **Q5.3.2** ✅ **RÉPONDU** : Délais entiers en samples
- **Q5.3.3** ✅ **RÉPONDU** : Oui, le canal LFE de sortie reçoit directement le canal LFE créé à l'étape 2

## 6. FORMATS ET PARAMÈTRES GLOBAUX

### 6.1 Formats d'entrée/sortie

- **Q6.1.1** ✅ **RÉPONDU** : Oui, ce sera un dropdown dans une UI
- **Q6.1.2** ✅ **RÉPONDU** : Idem
- **Q6.1.3** ✅ **RÉPONDU** : Ordre standard donné dans multichannel_layouts.py

### 6.2 Structure JSON

- **Q6.2.1** ✅ **RÉPONDU** : Oui, on peut partir là-dessus (structure proposée validée)
- **Q6.2.2** ⚠️ **TODO** : À voir, TODO

### 6.3 Traitement des erreurs

- **Q6.3.1** ✅ **RÉPONDU** : On détecte le nombre de canaux et on propose les formats possibles, on demande confirmation
- **Q6.3.2** ⚠️ **TODO** : À voir, TODO
- **Q6.3.3** ✅ **RÉPONDU** : On retourne une erreur et on arrête

## 7. DÉTAILS TECHNIQUES

### 7.1 Bibliothèques

- **Q7.1.1** ✅ **RÉPONDU** : scipy pour tout

### 7.2 Précision numérique

- **Q7.2.1** ✅ **RÉPONDU** : float32
- **Q7.2.2** ✅ **RÉPONDU** : Oui, on garde le niveau RMS

### 7.3 Performance

- **Q7.3.1** ✅ **RÉPONDU** : Non, pas de contrainte, c'est de la conversion, donc offline
- **Q7.3.2** ✅ **RÉPONDU** : Oui, traitement par blocs internes autorisé
