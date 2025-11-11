# Git Portable Installation Script
# Install Git portable version to current directory

$ErrorActionPreference = "Stop"
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8

Write-Host "=== Git Portable Installation Script ===" -ForegroundColor Cyan
Write-Host ""

# Get current directory
$currentDir = $PSScriptRoot
if (-not $currentDir) {
    $currentDir = Get-Location
}
$gitDir = Join-Path $currentDir "git-portable"
$downloadedFile = Join-Path $env:TEMP "PortableGit.7z.exe"

# Git portable download URL (latest stable version)
$gitUrl = "https://github.com/git-for-windows/git/releases/download/v2.43.0.windows.1/PortableGit-2.43.0-64-bit.7z.exe"

try {
    # Check if already installed
    $gitExe = Join-Path $gitDir "bin\git.exe"
    if (Test-Path $gitExe) {
        Write-Host "Git portable already installed!" -ForegroundColor Green
        Write-Host "Git path: $gitExe" -ForegroundColor Cyan
        & $gitExe --version
        Write-Host ""
        Write-Host "To reinstall, delete the git-portable directory first" -ForegroundColor Yellow
        exit 0
    }

    Write-Host "Step 1/3: Downloading Git portable..." -ForegroundColor Yellow
    Write-Host "Download URL: $gitUrl" -ForegroundColor Gray
    
    # Create temp directory if not exists
    if (-not (Test-Path $env:TEMP)) {
        New-Item -ItemType Directory -Path $env:TEMP -Force | Out-Null
    }
    
    # Download Git portable
    $ProgressPreference = 'Continue'
    try {
        Invoke-WebRequest -Uri $gitUrl -OutFile $downloadedFile -UseBasicParsing
    } catch {
        Write-Host "Download failed, trying alternative method..." -ForegroundColor Yellow
        $client = New-Object System.Net.WebClient
        $client.DownloadFile($gitUrl, $downloadedFile)
        $client.Dispose()
    }

    if (-not (Test-Path $downloadedFile)) {
        throw "Download failed: file does not exist"
    }

    $fileSize = (Get-Item $downloadedFile).Length / 1MB
    Write-Host "Download completed! File size: $([math]::Round($fileSize, 2)) MB" -ForegroundColor Green
    Write-Host ""

    Write-Host "Step 2/3: Extracting Git..." -ForegroundColor Yellow
    
    # Create target directory
    if (Test-Path $gitDir) {
        Remove-Item -Path $gitDir -Recurse -Force -ErrorAction SilentlyContinue
    }
    New-Item -ItemType Directory -Path $gitDir -Force | Out-Null

    # Method 1: Try using 7-Zip if installed
    $7zPath = Get-Command 7z -ErrorAction SilentlyContinue
    if ($7zPath) {
        Write-Host "Using 7-Zip to extract..." -ForegroundColor Gray
        & 7z x $downloadedFile "-o$gitDir" -y | Out-Null
    } 
    # Method 2: Run self-extracting executable
    else {
        Write-Host "Running Git self-extracting executable..." -ForegroundColor Gray
        Write-Host "Note: This may require user interaction" -ForegroundColor Yellow
        
        # Try silent extraction
        $arguments = "-o`"$gitDir`" -y"
        $process = Start-Process -FilePath $downloadedFile -ArgumentList $arguments -Wait -PassThru -NoNewWindow
        
        # If silent extraction failed, try interactive
        if (-not (Test-Path $gitExe)) {
            Write-Host "Silent extraction may have failed. Trying interactive mode..." -ForegroundColor Yellow
            Write-Host "Please select extraction path as: $gitDir" -ForegroundColor Yellow
            Start-Process -FilePath $downloadedFile -Wait
        }
    }

    # Verify installation
    Start-Sleep -Seconds 2
    if (-not (Test-Path $gitExe)) {
        throw "Extraction failed: git.exe not found. Please check $gitDir directory"
    }

    Write-Host "Extraction completed!" -ForegroundColor Green
    Write-Host ""

    Write-Host "Step 3/3: Verifying installation..." -ForegroundColor Yellow
    & $gitExe --version
    
    Write-Host ""
    Write-Host "=== Installation Successful! ===" -ForegroundColor Green
    Write-Host ""
    Write-Host "Git installation path: $gitDir" -ForegroundColor Cyan
    Write-Host "Git executable: $gitExe" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "Usage:" -ForegroundColor Yellow
    Write-Host "  1. Use full path:" -ForegroundColor White
    Write-Host "     $gitExe --version" -ForegroundColor Gray
    Write-Host ""
    Write-Host "  2. Add to current PowerShell session PATH:" -ForegroundColor White
    Write-Host "     `$env:PATH += `";$gitDir\bin;$gitDir\cmd`"" -ForegroundColor Gray
    Write-Host "     git --version" -ForegroundColor Gray
    Write-Host ""
    Write-Host "  3. Add to system PATH permanently (requires admin):" -ForegroundColor White
    $binPath = "$gitDir\bin"
    $cmdPath = "$gitDir\cmd"
    Write-Host "     [Environment]::SetEnvironmentVariable('Path', [Environment]::GetEnvironmentVariable('Path', 'User') + `";$binPath;$cmdPath`", 'User')" -ForegroundColor Gray

    # Clean up downloaded file
    if (Test-Path $downloadedFile) {
        Write-Host ""
        Write-Host "Cleaning up temporary files..." -ForegroundColor Gray
        Remove-Item $downloadedFile -Force -ErrorAction SilentlyContinue
    }

} catch {
    Write-Host ""
    Write-Host "=== Installation Failed ===" -ForegroundColor Red
    Write-Host "Error: $_" -ForegroundColor Red
    Write-Host "Error details: $($_.Exception.Message)" -ForegroundColor Red
    Write-Host ""
    Write-Host "Suggestions:" -ForegroundColor Yellow
    Write-Host "  1. Check internet connection" -ForegroundColor White
    Write-Host "  2. Manually download Git portable from: https://github.com/git-for-windows/git/releases" -ForegroundColor White
    Write-Host "  3. Or use winget to install: winget install --id Git.Git -e" -ForegroundColor White
    exit 1
}
