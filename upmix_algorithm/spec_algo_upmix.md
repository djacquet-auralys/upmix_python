Cette spec décris un algorithme d'upmix
la description complète est ici : C:\Users\Damien\Documents\Audiolift\Python\auralys_upmix\upmix_algorithm\brevet_upmix_auralys_v2.pdf

L'algo d'upmix prends en entrée :

- un signal multicanal (stéréo à minima, ou plusieurs cannaux) au format WAV
- son format d'entrée (voir C:\Users\Damien\Documents\Audiolift\Python\auralys_upmix\multichannel_layouts.py)
- son format de sortie (voir C:\Users\Damien\Documents\Audiolift\Python\auralys_upmix\multichannel_layouts.py)

un jeu de paramètres au format JSON
de ce type : C:\Users\Damien\Documents\Audiolift\Python\auralys_upmix\upmixparams.json
Les paramètres F_xover1 et F_LFE doivent être inclus dans le JSON.
"upmix_params": {
        "width": 0.18181818181818182,
        "slope": 500.0,
        "min_gain": -40.0,
        "attack": 1.0,
        "pan1": 0.9090909090909086,
        "gains1": [
          1.5482419980211148e-06,
          7.806699772540012e-08,
          1.4318764978203296e-08,
          1.1491320790990218e-08,
          2.0396427121950705e-08,
          4.036127511426092e-08,
          7.012061249303843e-08,
          1.0466938612659778e-07,
          1.364337993967741e-07,
          4.898979446815034e-07,
          2.5936263624672223e-06,
          0.0008043519280686413,
          0.6782657048512724,
          0.0,
          0.0,
          0.0
        ],
        "delays1": [
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0
        ],
        "release1": 186.3636363636363,
        "mute1": 0,
        "LF_gain1": 1.0,
        "pan2": 0.7272727272727267,
        "gains2": [
          1.067972805168175e-07,
          1.8077472550343473e-08,
          7.656800308294436e-09,
          8.837375605140511e-09,
          2.425159106066236e-08,
          6.797455386363144e-08,
          1.80281197357496e-07,
          4.697161155843636e-07,
          1.1996818099895032e-06,
          2.0625686075441945e-05,
          0.0004297691674682758,
          0.6782660079959171,
          0.0002250276458171932,
          0.0,
          0.0,
          0.0
        ],
        "delays2": [
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0
        ],
        "release2": 159.090909090909,
        "mute2": 0,
        "LF_gain2": 1.0,
        "pan3": 0.5454545454545451,
        "gains3": [
          1.4461018775025974e-08,
          5.4925829762406905e-09,
          4.616092140502406e-09,
          7.453647619082283e-09,
          3.1159220334139825e-08,
          1.2289913703028735e-07,
          5.138453448241725e-07,
          2.745295148229526e-06,
          2.1530771068895415e-05,
          0.014572546540104137,
          0.6781096039083319,
          0.00013736844445296856,
          1.402017935938934e-06,
          0.0,
          0.0,
          0.0
        ],
        "delays3": [
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0
        ],
        "release3": 131.81818181818176,
        "mute3": 0,
        "LF_gain3": 1.0,
        "pan4": 0.3636363636363634,
        "gains4": [
          4.292112562076311e-08,
          3.119733246037223e-08,
          5.03606339459895e-08,
          1.185486044274486e-07,
          8.62941348222902e-07,
          5.521466601264967e-06,
          4.466106226294265e-05,
          0.0007658153059387898,
          0.06469390211525534,
          0.6751652468346483,
          0.0033186216514473634,
          1.0746707112726633e-05,
          7.304723238334484e-07,
          0.0,
          0.0,
          0.0
        ],
        "delays4": [
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0
        ],
        "release4": 104.5454545454545,
        "mute4": 0,
        "LF_gain4": 1.0,
        "pan5": 0.18181818181818155,
        "gains5": [
          6.699488135969227e-09,
          8.81440961749293e-09,
          2.796192024573522e-08,
          1.0659788759225655e-07,
          2.0946905392555733e-06,
          4.2049472071494176e-05,
          0.0020003935587377436,
          0.6607752285687667,
          0.15302648937402905,
          2.9500763507872047e-05,
          3.3630307446038493e-06,
          1.5725665133856735e-07,
          3.169258383151357e-08,
          0.0,
          0.0,
          0.0
        ],
        "delays5": [
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0
        ],
        "release5": 77.27272727272722,
        "mute5": 0,
        "LF_gain5": 1.0,
        "pan6": -2.220446049250313e-16,
        "gains6": [
          2.7752011009934096e-09,
          6.463498481796714e-09,
          4.226922885610109e-08,
          2.845180723385254e-07,
          2.3746123175885608e-05,
          0.00416876379774682,
          0.6782394000131813,
          0.004353109355547615,
          3.0362745113827404e-05,
          2.4294437729440194e-07,
          7.09051005373663e-08,
          1.1369752439916531e-08,
          4.600609081194329e-09,
          0.0,
          0.0,
          0.0
        ],
        "delays6": [
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0
        ],
        "release6": 50.000000000000036,
        "mute6": 0,
        "LF_gain6": 1.0,
        "pan7": -0.181818181818182,
        "gains7": [
          1.4861932171385924e-08,
          7.285679759480902e-08,
          1.5889197747074367e-06,
          3.2246527606271304e-05,
          0.06411967981959964,
          0.6752250047505434,
          0.0022069033951584867,
          3.8971032563685855e-05,
          2.137980621047949e-06,
          8.374746882057981e-08,
          3.755140776302392e-08,
          1.243940375696952e-08,
          8.51626043666614e-09,
          0.0,
          0.0,
          0.0
        ],
        "delays7": [
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0
        ],
        "release7": 77.2727272727273,
        "mute7": 0,
        "LF_gain7": 1.0,
        "pan8": -0.36363636363636376,
        "gains8": [
          1.1707749452186203e-07,
          1.9589218851715387e-06,
          0.00082941740572687,
          0.6777386452321453,
          0.026731920557991867,
          0.00024163037341257877,
          1.5013359506997895e-05,
          1.784068642286124e-06,
          3.0688101911259926e-07,
          3.724716619212931e-08,
          2.3782726079698277e-08,
          1.508810828710513e-08,
          1.755367746286183e-08,
          0.0,
          0.0,
          0.0
        ],
        "delays8": [
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0
        ],
        "release8": 104.54545454545456,
        "mute8": 0,
        "LF_gain8": 1.0,
        "pan9": -0.5454545454545455,
        "gains9": [
          5.929833140166817e-07,
          7.46745448766179e-05,
          0.6782589235173903,
          0.003136890230049787,
          1.938205242231523e-05,
          2.239024798593351e-06,
          4.5741192140543894e-07,
          1.1309269568276713e-07,
          3.2624700176299445e-08,
          8.03759025111073e-09,
          7.013999080601357e-09,
          8.234376336135013e-09,
          1.667653651345598e-08,
          0.0,
          0.0,
          0.0
        ],
        "delays9": [
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0
        ],
        "release9": 131.8181818181818,
        "mute9": 0,
        "LF_gain9": 1.0,
        "pan10": -0.7272727272727273,
        "gains10": [
          9.219489717714633e-05,
          0.6782659332241707,
          0.0005729199459260947,
          2.1297314557629838e-05,
          2.2172718316306926e-06,
          7.628265307176365e-07,
          2.9213099204801735e-07,
          1.0953178537319326e-07,
          4.2634194772941293e-08,
          1.7240926075419352e-08,
          2.003894516935963e-08,
          4.4215735471964874e-08,
          1.734790399426215e-07,
          0.0,
          0.0,
          0.0
        ],
        "delays10": [
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0
        ],
        "release10": 159.0909090909091,
        "mute10": 0,
        "LF_gain10": 1.0,
        "pan11": -0.9090909090909091,
        "gains11": [
          0.6782466634584766,
          0.005145557861357458,
          9.271673207838474e-06,
          1.7730478352113317e-06,
          6.960876732072575e-07,
          4.845130458534044e-07,
          3.0970262454378066e-07,
          1.7203692695078013e-07,
          9.04695167146459e-08,
          6.035450001812133e-08,
          9.537013504108651e-08,
          4.494600522766808e-07,
          4.515353422211303e-06,
          0.0,
          0.0,
          0.0
        ],
        "delays11": [
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0
        ],
        "release11": 186.36363636363635,
        "mute11": 0,
        "LF_gain11": 1.0
      }
    }

le format du signal multicanal de sortie
(voir C:\Users\Damien\Documents\Audiolift\Python\auralys_upmix\multichannel_layouts.py)

il sort un signal multicanal dans le format de sortie au format WAV

- même fréquence d'échantillonage que l'entrée
- même bitrate que l'entrée
- traitement effectué sur fichier complet (pas en streaming)

il y a plusieurs étapes de traitement :

1. Crossovers stéréo  (pas présent dans le pdf)

- 2 IIR biquad en cascade (ordre 4) pour avoir une coupure à -6 dB à la fréquence de coupure
- Formules : W3C Audio EQ Cookbook (<https://www.w3.org/TR/audio-eq-cookbook/#formulae>)
- Types : HPF (High Pass Filter) et LPF (Low Pass Filter) selon le cookbook
- Fréquence variable F_xover1 (défaut: 150 Hz), Q = 0.707
- Les 2 biquads ont la même fréquence de coupure

Pour stéréo :

- on obtient les signaux L_lowfreq et R_lowfreq, L_highfreq et R_highfreq
- L_lowfreq et R_lowfreq sont sommés en un signal LF_mono1 = (L_lowfreq + R_lowfreq) * 0.707
- (on peut aussi faire la sommation mono d'abord puis un seul passe bas avec 2 biquads)

Pour généraliser au multicanal :

- pour obtenir LF_mono1, on somme à puissance constante : LF_mono1 = sqrt(sum(L_lowfreq² + R_lowfreq² + ...))
- tous les canaux du signal d'entrée SAUF canaux LFE, tous filtrés de manière identique par ce crossover
- On garde également tous les signaux non LFE filtrés (les fréquences hautes) => s1_HF, s2_HF...

2. création canal LFE

- Détection LFE : par le label dans multichannel_layouts.py
- Si LFE déjà présent : utiliser directement
  - Si plusieurs LFE : sommer en mono (TODO: réfléchir à ce qu'il faut faire par la suite)
- Si LFE absent :
  - Sommation mono à puissance constante : sqrt(sum(s1² + s2² + ...)) de tous les canaux SAUF LFE (TODO: réfléchir)
  - Puis LP filter : 2 biquads en cascade (ordre 4), Q = 0.707, fréquence F_LFE (défaut: 120 Hz)

à ce stade on à :
autant de cannaux que les canaux non LFE, mais seulement les hautes fréquences => L_highfreq et R_highfreq pour de la stéréo, sinon on les appelle s1_HF, s2_HF...
1 canal LFE
1 canal LF_mono1

3. upmix fréquenciel à partir des s1_HF, s2_HF...

on crée un certain nombre de cannaux, dépendant du système de destination (nombre de HP hors LFE) comme dans
C:\Users\Damien\Documents\Audiolift\Python\auralys_upmix\auralys_upmix_poc_ui.py
nb_source = np.min([nb_spk + (1- (nb_spk % 2)),max_sources])

pour chaque canal à créer, on associe les paramètres d'extraction suivant:

width
slope
min_gain
attack
release
pan
gains
delays
LF_gain

on va travailler dans le domaine fréquentiel :

- STFT de 128 points par défaut (nwin = taille fenêtre, nfreq = nombre bins = nwin/2+1)
- Paramètres réglables dans le script (nfft, overlap)
- hop_size = nwin * 0.25 = 32 samples (pour nwin=128)
- Fenêtre : sqrt(hann) pour analyse ET synthèse (fenêtre duale pour reconstruction parfaite)

le traitement frequentiel est le suivant :

en sortie de la stft complexe, on a S1, S2, S3(t,f)

estimation de panning :

- Utiliser re.compute_re_model pour obtenir l'estimation de panning
- Les gains sont remplacés par les modules des STFT à chaque fréquence (|S1|, |S2|, ...)
- On considère les délais nuls dans le modèle par vecteurs d'énergie
- On obtient un vecteur d'énergie (x, y) pour chaque fréquence
- On calcule l'angle : atan2(y, x)
- On doit connaître les coordonnées des canaux d'entrée (depuis le format d'entrée)
- Normalisation : par l'ouverture maximum du setup de départ
  - 60° pour la stéréo
  - 360° pour multicanal (pas de calcul depuis azimuts réels)
- Résultat : à chaque fenêtre et pour chaque fréquence, un indice entre -1 et 1

extraction :

- LUT masque : calcul pour chaque canal à créer
  - Formule : y = 10^(max(min(SLOPE * (W/2 - abs(x - PAN)), 0), min_gain) / 20.0)
  - Résolution : 200 points entre -1 et 1
  - Interpolation : linéaire
- Lissage du gain :
  1. Sur l'axe fréquentiel : blur triangulaire ±1 bin (3 bins, linéaire décroissant)
     - Appliqué à l'intérieur d'une même fenêtre temporelle
     - Hors DC et Nyquist
  2. En temporel : rampsmooth (rampup attack, rampdown release)
     - Unités : en frames STFT
     - Algorithme : voir code de référence (gen~ RNBO)
     - Freeze si power < 1e-6
     - Doublage du release pour bin 0 et Nyquist
- Sélection signal : parmi S1, S2, S3, choisir le signal le plus proche de l'angle trouvé
  - Angle déterminé depuis le format d'entrée
- Application : multiplication complexe S_selected * gain_lissé
- STFT inverse : avec overlap-add et normalisation
  - Fenêtre duale : sqrt(hann)
  - Pas de modulation ou variation d'amplitude significative

4. ajout LF_mono1 (pas présent dans le pdf)

- Délai de latence : 256 samples pour nfft = 128 (valeur utilisée dans MaxMSP)
  - Nécessaire pour aligner temporellement et en phase le signal généré par le traitement fréquentiel avec le signal original
  - Application : en samples (délai entier)
- Application du gain :
  - LF_gain en dB
  - Le signal LF_mono1 retardé est sommé à CHAQUE signal obtenu à l'issue de l'extraction
  - Chaque source a son propre LF_gain[i]

5. respatialisation

- Coordonnées HP destination : données par le format de destination (multichannel_layouts.py)
- Paramètres de placement (dans JSON) :
  - audience_bary : origine du système [0.0, 0.0]
  - panorama_center : speaker central ou position équivalente à panning = 0
  - panorama_width : largeur du panorama en degrés
- Calcul gains de spatialisation :
  - Si gains/delays déjà calculés dans JSON : appliquer directement
  - Sinon : utiliser TDAP pour calculer
  - Délais en ms dans JSON, convertis en samples entiers pour application
- Application aux canaux de sortie :
  - Pour chaque HP de destination : somme de toutes les sources avec leurs gains/délais respectifs
  - output_channel[i] = sum(source[j] *gains[j][i]* delay(delays[j][i]))
  - Délais : entiers en samples (pas de délai fractionnaire)
- Canal LFE : appliqué directement au canal LFE de sortie (créé à l'étape 2)

TODO: Réfléchir sur le placement des sources - soit fichiers JSON par configuration, soit déjà défini dans JSON (le mieux), soit recalcul à chaque fois

---

## DÉTAILS TECHNIQUES

### Bibliothèques

- Utiliser scipy pour tout (filtres IIR, STFT, FFT, lecture/écriture WAV)

### Précision numérique

- Format interne : float32
- Normalisation : préserver le niveau RMS d'entrée

### Performance

- Pas de contrainte de performance (traitement offline)
- Traitement par blocs internes autorisé (même si fichier complet en entrée)

### Formats d'entrée/sortie

- Formats : spécifiés par dropdown dans UI (stereo, 5.1, 7.1, etc. depuis multichannel_layouts.py)
- Ordre des canaux : ordre standard donné dans multichannel_layouts.py
- Détection automatique : si nombre de canaux WAV ne correspond pas au format déclaré, proposer les formats possibles et demander confirmation

### Structure JSON

Structure complète attendue :

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

### Gestion des erreurs

- Signaux vides, NaN, Inf : retourner une erreur et arrêter
- JSON avec plus de sources que nécessaire : TODO à définir
- Paramètres optionnels vs obligatoires : TODO à définir

### Points à clarifier / TODOs

1. **LFE multiple** : Si plusieurs LFE, on somme en mono. TODO: réfléchir à ce qu'il faut faire par la suite
2. **Canaux pour création LFE** : Tous sauf LFE. TODO: réfléchir à ce qu'il faut faire par la suite  
3. **Placement sources** : TODO: Réfléchir - soit fichiers JSON par configuration, soit déjà défini (le mieux), soit recalcul à chaque fois
4. **Paramètres optionnels JSON** : TODO à définir
5. **JSON avec plus de sources** : TODO à définir
6. **Fenêtre STFT** : À vérifier lors de l'implémentation (à priori sqrt(hann))
7. **Unités attack/release** : À confirmer (à priori en frames STFT)
8. **JSON plus clair** : Un JSON plus clair sera fourni avec les paramètres de respatialisation
