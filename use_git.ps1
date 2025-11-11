# Quick Git PATH Setup
# Run this in your PowerShell: . .\use_git.ps1

$ErrorActionPreference = "Continue"

# Use current working directory (more reliable with special characters in path)
$currentDir = Get-Location

# Git paths (relative to current directory)
$gitBin = Join-Path $currentDir "git-portable\bin"
$gitCmd = Join-Path $currentDir "git-portable\cmd"
$gitExe = Join-Path $gitBin "git.exe"

# Check if Git exists
if (-not (Test-Path $gitExe)) {
    Write-Host "ERROR: Git not found at $gitExe" -ForegroundColor Red
    Write-Host "Current directory: $currentDir" -ForegroundColor Yellow
    Write-Host "Please ensure you are in the project root directory." -ForegroundColor Yellow
    Write-Host "If Git is not installed, run: .\install_git.ps1" -ForegroundColor Yellow
    return
}

# Add to PATH using array method (more reliable)
$pathArray = $env:PATH -split ';' | Where-Object { $_ -ne '' }
$needsUpdate = $false

if ($pathArray -notcontains $gitBin) {
    $pathArray += $gitBin
    $needsUpdate = $true
}

if ($pathArray -notcontains $gitCmd) {
    $pathArray += $gitCmd
    $needsUpdate = $true
}

if ($needsUpdate) {
    $env:PATH = $pathArray -join ';'
    Write-Host "Git added to PATH" -ForegroundColor Green
} else {
    Write-Host "Git is already in PATH" -ForegroundColor Green
}

# Test Git
Write-Host ""
Write-Host "Testing Git..." -ForegroundColor Yellow
& $gitExe --version
if ($LASTEXITCODE -eq 0) {
    Write-Host ""
    Write-Host "SUCCESS! You can now use 'git' command." -ForegroundColor Green
    Write-Host "Example: git init" -ForegroundColor Cyan
} else {
    Write-Host "WARNING: Git test failed" -ForegroundColor Yellow
}

