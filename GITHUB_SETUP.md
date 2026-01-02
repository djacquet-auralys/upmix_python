# Guide : Créer le dépôt GitHub et connecter le projet

## ✅ Étape 1 : Dépôt Git local créé

Le dépôt Git local a été initialisé et le premier commit a été créé avec succès.

## 📋 Étape 2 : Créer le dépôt sur GitHub

### Option A : Via l'interface web GitHub

1. **Aller sur GitHub** : <https://github.com>
2. **Se connecter** à votre compte
3. **Cliquer sur le bouton "+"** en haut à droite → **"New repository"**
4. **Remplir les informations** :
   - **Repository name** : `auralys_upmix` (ou le nom de votre choix)
   - **Description** : "Algorithme d'upmix audio pour conversion stéréo/multicanal vers surround"
   - **Visibilité** : Public ou Private (selon votre choix)
   - **⚠️ IMPORTANT** : **NE PAS** cocher "Initialize this repository with a README" (on a déjà un README)
   - **NE PAS** ajouter `.gitignore` ou une licence (on a déjà un `.gitignore`)
5. **Cliquer sur "Create repository"**

### Option B : Via GitHub CLI (si installé)

```bash
gh repo create auralys_upmix --public --description "Algorithme d'upmix audio"
```

## 🔗 Étape 3 : Connecter le dépôt local à GitHub

Une fois le dépôt créé sur GitHub, vous verrez une page avec des instructions.

### Si vous créez un nouveau dépôt (sans README)

GitHub vous donnera des commandes similaires à :

```bash
git remote add origin https://github.com/VOTRE_USERNAME/auralys_upmix.git
git branch -M main
git push -u origin main
```

### Commandes à exécuter dans PowerShell

**Remplacez `VOTRE_USERNAME` par votre nom d'utilisateur GitHub** :

```powershell
cd "c:\Users\Damien\Documents\Audiolift\Python\auralys_upmix"

# Ajouter le remote GitHub
git remote add origin https://github.com/VOTRE_USERNAME/auralys_upmix.git

# Renommer la branche principale en 'main' (si nécessaire)
git branch -M main

# Pousser le code vers GitHub
git push -u origin main
```

## 🔐 Étape 4 : Authentification GitHub

Si c'est la première fois que vous poussez vers GitHub depuis cette machine, vous devrez vous authentifier :

### Option A : Token d'accès personnel (recommandé)

1. **Créer un token** : <https://github.com/settings/tokens>
   - Cliquer sur "Generate new token (classic)"
   - Donner un nom (ex: "auralys_upmix")
   - Cocher `repo` (accès complet aux dépôts)
   - Cliquer sur "Generate token"
   - **⚠️ Copier le token immédiatement** (il ne sera plus visible après)

2. **Utiliser le token** :
   - Quand Git vous demande votre mot de passe, utilisez le **token** au lieu du mot de passe
   - Ou utilisez l'URL avec le token :

   ```powershell
   git remote set-url origin https://VOTRE_TOKEN@github.com/VOTRE_USERNAME/auralys_upmix.git
   ```

### Option B : GitHub CLI (plus simple)

```bash
gh auth login
```

## ✅ Étape 5 : Vérification

Après le push, vérifiez que tout est bien sur GitHub :

```powershell
git remote -v
```

Vous devriez voir :

```
origin  https://github.com/VOTRE_USERNAME/auralys_upmix.git (fetch)
origin  https://github.com/VOTRE_USERNAME/auralys_upmix.git (push)
```

## 📝 Étape 6 : Mettre à jour le README (optionnel)

Une fois le dépôt créé, vous pouvez mettre à jour le README.md avec :

- L'URL du dépôt GitHub
- Les instructions de contribution
- La licence
- Les badges (si souhaité)

## 🚀 Commandes Git utiles pour la suite

```powershell
# Voir l'état des fichiers
git status

# Ajouter des fichiers modifiés
git add .

# Créer un commit
git commit -m "Description des changements"

# Pousser vers GitHub
git push

# Récupérer les changements depuis GitHub
git pull

# Voir l'historique
git log --oneline
```

## 🆘 En cas de problème

### Erreur : "remote origin already exists"

```powershell
git remote remove origin
git remote add origin https://github.com/VOTRE_USERNAME/auralys_upmix.git
```

### Erreur : "failed to push some refs"

```powershell
git pull origin main --allow-unrelated-histories
git push -u origin main
```

### Changer l'URL du remote

```powershell
git remote set-url origin https://github.com/VOTRE_USERNAME/auralys_upmix.git
```
