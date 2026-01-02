# Guide de Configuration de l'Environnement Virtuel

Ce guide vous explique comment configurer un environnement virtuel Python pour ce projet dans Cursor.

## 📋 Prérequis

- Python 3.8 ou supérieur installé
- Cursor avec l'extension Python installée

## 🚀 Configuration étape par étape

### Étape 1 : Créer l'environnement virtuel

Ouvrez un terminal dans Cursor (`Ctrl+`` ou `Terminal` → `New Terminal`) et exécutez :

```powershell
# Créer l'environnement virtuel dans le dossier .venv
python -m venv .venv
```

**Note :** Si vous avez plusieurs versions de Python, utilisez :
```powershell
python3 -m venv .venv
# ou
py -3.11 -m venv .venv  # pour une version spécifique
```

### Étape 2 : Activer l'environnement virtuel

**Sur Windows PowerShell :**
```powershell
.venv\Scripts\Activate.ps1
```

**Sur Windows CMD :**
```cmd
.venv\Scripts\activate.bat
```

**Sur Linux/Mac :**
```bash
source .venv/bin/activate
```

**⚠️ Si vous obtenez une erreur d'exécution de script sur PowerShell :**
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```
Puis réessayez d'activer l'environnement.

### Étape 3 : Mettre à jour pip

```powershell
python -m pip install --upgrade pip
```

### Étape 4 : Installer les dépendances

```powershell
# Installer toutes les dépendances du projet
pip install -r requirements.txt
```

**Ou si vous travaillez uniquement sur flask_mockup :**
```powershell
pip install -r flask_mockup/requirements.txt
```

### Étape 5 : Vérifier l'installation

```powershell
# Vérifier que les packages sont installés
pip list

# Tester l'import de numpy et matplotlib
python -c "import numpy; import matplotlib; print('✅ Dépendances installées avec succès!')"
```

## 🔧 Configuration dans Cursor

### Sélectionner l'interpréteur Python

1. Appuyez sur `Ctrl+Shift+P` pour ouvrir la palette de commandes
2. Tapez : `Python: Select Interpreter`
3. Choisissez l'interpréteur dans `.venv\Scripts\python.exe`

**Ou :**
- Cliquez sur l'indicateur Python en bas à droite de Cursor
- Sélectionnez `.venv\Scripts\python.exe`

### Vérifier que Cursor utilise le bon interpréteur

- En bas à droite de Cursor, vous devriez voir : `Python 3.x.x ('.venv': venv)`
- Si ce n'est pas le cas, suivez l'étape ci-dessus

## ✅ Vérification finale

Créez un fichier de test `test_env.py` :

```python
import sys
import numpy as np
import matplotlib.pyplot as plt

print(f"Python: {sys.version}")
print(f"NumPy: {np.__version__}")
print(f"Matplotlib: {plt.__version__}")
print(f"Environnement virtuel: {sys.prefix}")
```

Exécutez-le :
```powershell
python test_env.py
```

Vous devriez voir que `sys.prefix` pointe vers `.venv`.

## 🎯 Utilisation quotidienne

### Activer l'environnement à chaque session

**Option 1 : Automatique (recommandé)**
- Cursor détecte automatiquement `.venv` si configuré dans `settings.json`
- L'environnement s'active automatiquement dans le terminal intégré

**Option 2 : Manuel**
- Ouvrez un terminal dans Cursor
- Exécutez : `.venv\Scripts\Activate.ps1`

### Installer de nouveaux packages

```powershell
# Activer l'environnement (si pas déjà fait)
.venv\Scripts\Activate.ps1

# Installer un package
pip install nom_du_package

# Mettre à jour requirements.txt
pip freeze > requirements.txt
```

### Désactiver l'environnement

```powershell
deactivate
```

## 🐛 Dépannage

### Problème : "python n'est pas reconnu"
- Vérifiez que Python est dans votre PATH
- Utilisez `py` au lieu de `python` sur Windows

### Problème : Cursor ne détecte pas l'environnement virtuel
1. Fermez et rouvrez Cursor
2. Vérifiez que `.venv` existe dans le dossier du projet
3. Sélectionnez manuellement l'interpréteur (`Ctrl+Shift+P` → `Python: Select Interpreter`)

### Problème : Les imports ne fonctionnent pas
1. Vérifiez que l'environnement virtuel est activé dans le terminal
2. Vérifiez que les packages sont installés : `pip list`
3. Réinstallez les dépendances : `pip install -r requirements.txt`

### Problème : Erreur d'exécution de script PowerShell
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

## 📝 Notes importantes

- **Ne commitez jamais** le dossier `.venv/` dans Git (il devrait être dans `.gitignore`)
- **Commitez** `requirements.txt` pour partager les dépendances
- L'environnement virtuel est spécifique à chaque projet
- Vous pouvez avoir plusieurs environnements virtuels pour différents projets

## 🔗 Ressources utiles

- [Documentation Python venv](https://docs.python.org/3/library/venv.html)
- [Extension Python pour VS Code/Cursor](https://marketplace.visualstudio.com/items?itemName=ms-python.python)




