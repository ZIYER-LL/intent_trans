# Git Environment Setup
# Source this file in PowerShell: . .\git_env.ps1

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$gitBinPath = Join-Path $scriptDir "git-portable\bin"
$gitCmdPath = Join-Path $scriptDir "git-portable\cmd"

if (Test-Path (Join-Path $gitBinPath "git.exe")) {
    if ($env:PATH -notlike "*$gitBinPath*") {
        $env:PATH = "$env:PATH;$gitBinPath;$gitCmdPath"
        Write-Host "Git added to PATH" -ForegroundColor Green
    }
    git --version
} else {
    Write-Host "Git not found. Please run install_git.ps1 first." -ForegroundColor Red
}

