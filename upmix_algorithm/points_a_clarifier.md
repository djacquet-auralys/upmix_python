# Points à Clarifier / TODOs - Résumé

## TODOs Identifiés

### 1. LFE Multiple (Q2.1.2)

**Contexte** : Si plusieurs canaux LFE existent (ex: 10.2 avec LFE1 et LFE2)
**Action actuelle** : On somme en mono
**TODO** : Réfléchir à ce qu'il faut faire par la suite
**Priorité** : Moyenne

### 2. Canaux pour Création LFE (Q2.2.2)

**Contexte** : Quels canaux inclure dans la somme pour créer le LFE si absent ?
**Action actuelle** : Tous sauf LFE
**TODO** : Réfléchir à ce qu'il faut faire par la suite (peut-être seulement canaux frontaux ?)
**Priorité** : Moyenne

### 3. Placement des Sources (Q5.1.2)

**Contexte** : Si les positions des sources sont déjà définies (via pan/gains/delays), faut-il recalculer ?
**Options** :

- Fichiers JSON par configuration d'arrivée
- Déjà défini dans JSON (le mieux)
- Recalcul à chaque fois
**TODO** : Décider de l'approche
**Priorité** : Haute (impact sur structure JSON)

### 4. Paramètres Optionnels JSON (Q6.2.2)

**Contexte** : Quels paramètres ont des valeurs par défaut vs obligatoires ?
**TODO** : Définir la liste des paramètres optionnels et leurs valeurs par défaut
**Priorité** : Moyenne

### 5. JSON avec Plus de Sources (Q6.3.2)

**Contexte** : Que faire si le JSON contient des paramètres pour plus de sources que nécessaire ?
**TODO** : Définir le comportement (ignorer les extras ? erreur ?)
**Priorité** : Basse

## Points à Vérifier lors de l'Implémentation

### 1. Fenêtre STFT (Q3.3.3)

**Statut** : À vérifier
**Hypothèse** : sqrt(hann) pour avoir la même fenêtre à l'entrée et à la sortie
**Action** : Vérifier lors de l'implémentation du STFT

### 2. Unités Attack/Release (Q3.2.3)

**Statut** : À confirmer
**Hypothèse** : En frames STFT
**Action** : Confirmer lors de l'implémentation du lissage temporel

### 3. JSON Plus Clair (Q5.1.1)

**Statut** : En attente
**Action** : Attendre le JSON plus clair avec les paramètres de respatialisation

## Questions Résolues ✅

Toutes les autres questions ont été résolues et intégrées dans la spécification détaillée (`spec_detailed.md`) et la spécification principale (`spec_algo_upmix`).


