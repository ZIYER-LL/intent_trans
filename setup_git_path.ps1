# Setup Git PATH for current PowerShell session
# Run this script to add Git portable to your PATH

$ErrorActionPreference = "Stop"

# Get current directory
$currentDir = $PSScriptRoot
if (-not $currentDir) {
    $currentDir = Get-Location
}

# Git portable paths
$gitBinPath = Join-Path $currentDir "git-portable\bin"
$gitCmdPath = Join-Path $currentDir "git-portable\cmd"

# Check if Git is installed
$gitExe = Join-Path $gitBinPath "git.exe"
if (-not (Test-Path $gitExe)) {
    Write-Host "Error: Git portable not found at $gitBinPath" -ForegroundColor Red
    Write-Host "Please run install_git.ps1 first to install Git." -ForegroundColor Yellow
    exit 1
}

# Add to PATH
Write-Host "Adding Git to PATH..." -ForegroundColor Yellow

# Get current PATH
$currentPath = $env:PATH

# Check if already in PATH
if ($currentPath -like "*$gitBinPath*" -or $currentPath -like "*$gitCmdPath*") {
    Write-Host "Git is already in PATH." -ForegroundColor Green
} else {
    # Add Git paths to PATH
    $env:PATH = "$currentPath;$gitBinPath;$gitCmdPath"
    Write-Host "Git added to PATH successfully!" -ForegroundColor Green
}

# Verify Git works
Write-Host ""
Write-Host "Verifying Git installation..." -ForegroundColor Yellow
try {
    $gitVersion = & git --version 2>&1
    Write-Host $gitVersion -ForegroundColor Green
    Write-Host ""
    Write-Host "Git is now available! You can use 'git' command directly." -ForegroundColor Green
    Write-Host ""
    Write-Host "Note: This PATH setting is only for the current PowerShell session." -ForegroundColor Yellow
    Write-Host "To make it permanent, run:" -ForegroundColor Yellow
    Write-Host "  [Environment]::SetEnvironmentVariable('Path', [Environment]::GetEnvironmentVariable('Path', 'User') + `";$gitBinPath;$gitCmdPath`", 'User')" -ForegroundColor Gray
} catch {
    Write-Host "Error: Git command failed" -ForegroundColor Red
    Write-Host $_.Exception.Message -ForegroundColor Red
    exit 1
}
