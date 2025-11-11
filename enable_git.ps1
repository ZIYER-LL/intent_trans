# Quick script to enable Git in current PowerShell session
# Usage: . .\enable_git.ps1

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$gitBinPath = Join-Path $scriptDir "git-portable\bin"
$gitCmdPath = Join-Path $scriptDir "git-portable\cmd"

if (Test-Path (Join-Path $gitBinPath "git.exe")) {
    # Add to PATH if not already present
    if ($env:PATH -notlike "*$gitBinPath*") {
        $env:PATH = "$gitBinPath;$gitCmdPath;$env:PATH"
        Write-Host "[OK] Git added to PATH" -ForegroundColor Green
    } else {
        Write-Host "[INFO] Git already in PATH" -ForegroundColor Yellow
    }
    
    # Show Git version
    Write-Host "`nGit version: " -NoNewline
    & "$gitBinPath\git.exe" --version
    
    Write-Host "`nYou can now use 'git' command directly!" -ForegroundColor Green
} else {
    Write-Host "[ERROR] Git not found at $gitBinPath" -ForegroundColor Red
    Write-Host "Please run install_git.ps1 first to install Git." -ForegroundColor Yellow
}

