# Script de configuration automatique de l'environnement virtuel
# Usage: .\setup_venv.ps1

Write-Host "🚀 Configuration de l'environnement virtuel Python..." -ForegroundColor Cyan

# Vérifier que Python est installé
try {
    $pythonVersion = python --version 2>&1
    Write-Host "✅ Python détecté: $pythonVersion" -ForegroundColor Green
} catch {
    Write-Host "❌ Erreur: Python n'est pas installé ou pas dans le PATH" -ForegroundColor Red
    Write-Host "   Essayez d'utiliser 'py' au lieu de 'python'" -ForegroundColor Yellow
    exit 1
}

# Créer l'environnement virtuel
if (Test-Path ".venv") {
    Write-Host "⚠️  Le dossier .venv existe déjà" -ForegroundColor Yellow
    $response = Read-Host "Voulez-vous le recréer? (o/N)"
    if ($response -eq "o" -or $response -eq "O") {
        Remove-Item -Recurse -Force .venv
        Write-Host "🗑️  Ancien environnement virtuel supprimé" -ForegroundColor Yellow
    } else {
        Write-Host "ℹ️  Utilisation de l'environnement virtuel existant" -ForegroundColor Blue
    }
}

if (-not (Test-Path ".venv")) {
    Write-Host "📦 Création de l'environnement virtuel..." -ForegroundColor Cyan
    python -m venv .venv
    if ($LASTEXITCODE -ne 0) {
        Write-Host "❌ Erreur lors de la création de l'environnement virtuel" -ForegroundColor Red
        exit 1
    }
    Write-Host "✅ Environnement virtuel créé" -ForegroundColor Green
}

# Activer l'environnement virtuel
Write-Host "🔌 Activation de l'environnement virtuel..." -ForegroundColor Cyan
& .venv\Scripts\Activate.ps1

if ($LASTEXITCODE -ne 0) {
    Write-Host "⚠️  Erreur d'activation. Essayez d'exécuter:" -ForegroundColor Yellow
    Write-Host "   Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser" -ForegroundColor Yellow
    exit 1
}

# Mettre à jour pip
Write-Host "⬆️  Mise à jour de pip..." -ForegroundColor Cyan
python -m pip install --upgrade pip --quiet
Write-Host "✅ pip mis à jour" -ForegroundColor Green

# Installer les dépendances
if (Test-Path "requirements.txt") {
    Write-Host "📥 Installation des dépendances depuis requirements.txt..." -ForegroundColor Cyan
    pip install -r requirements.txt
    if ($LASTEXITCODE -eq 0) {
        Write-Host "✅ Dépendances installées avec succès!" -ForegroundColor Green
    } else {
        Write-Host "⚠️  Certaines dépendances n'ont pas pu être installées" -ForegroundColor Yellow
    }
} else {
    Write-Host "⚠️  Fichier requirements.txt non trouvé" -ForegroundColor Yellow
    Write-Host "   Installation des dépendances de base..." -ForegroundColor Cyan
    pip install numpy matplotlib
}

# Vérification
Write-Host "`n🔍 Vérification de l'installation..." -ForegroundColor Cyan
python -c "import sys; import numpy; import matplotlib; print(f'✅ Python {sys.version.split()[0]}'); print(f'✅ NumPy {numpy.__version__}'); print(f'✅ Matplotlib {matplotlib.__version__}'); print(f'✅ Environnement: {sys.prefix}')"

Write-Host "`n✨ Configuration terminée!" -ForegroundColor Green
Write-Host "`n📝 Prochaines étapes:" -ForegroundColor Cyan
Write-Host "   1. Dans Cursor: Ctrl+Shift+P → 'Python: Select Interpreter'" -ForegroundColor White
Write-Host "   2. Choisissez: .venv\Scripts\python.exe" -ForegroundColor White
Write-Host "   3. Ou utilisez le terminal intégré (l'env sera activé automatiquement)" -ForegroundColor White
Write-Host "`n💡 Pour activer manuellement: .venv\Scripts\Activate.ps1" -ForegroundColor Yellow




