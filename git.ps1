# Git wrapper script - Use this instead of direct git command
# Usage: .\git.ps1 init
#        .\git.ps1 --version
#        .\git.ps1 status

param(
    [Parameter(ValueFromRemainingArguments=$true)]
    [string[]]$Arguments
)

$gitExe = Join-Path $PSScriptRoot "git-portable\bin\git.exe"

if (-not (Test-Path $gitExe)) {
    Write-Host "ERROR: Git not found. Please run install_git.ps1 first." -ForegroundColor Red
    exit 1
}

& $gitExe $Arguments

