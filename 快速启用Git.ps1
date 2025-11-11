# 快速启用 Git - 在当前 PowerShell 会话中添加 Git 到 PATH
# 使用方法: 在 PowerShell 中运行: . .\快速启用Git.ps1

# 获取当前脚本所在目录（使用相对路径，避免路径编码问题）
$scriptPath = $MyInvocation.MyCommand.Path
if (-not $scriptPath) {
    $scriptPath = $PSScriptRoot
}
if (-not $scriptPath) {
    $scriptPath = Split-Path -Parent $MyInvocation.MyCommand.Path
}

# 使用相对路径（从当前工作目录）
$currentDir = Get-Location
$gitBinPath = Join-Path $currentDir "git-portable\bin"
$gitCmdPath = Join-Path $currentDir "git-portable\cmd"

# 检查 Git 是否存在
$gitExe = Join-Path $gitBinPath "git.exe"
if (-not (Test-Path $gitExe)) {
    Write-Host "错误: 找不到 Git!" -ForegroundColor Red
    Write-Host "Git 路径: $gitExe" -ForegroundColor Yellow
    Write-Host "当前目录: $currentDir" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "请确保:" -ForegroundColor Yellow
    Write-Host "  1. 你在项目根目录 (intent-trans)" -ForegroundColor White
    Write-Host "  2. git-portable 目录存在" -ForegroundColor White
    Write-Host "  3. 如果不存在，请运行: .\install_git.ps1" -ForegroundColor White
    return
}

# 添加到 PATH
$currentPath = $env:PATH
$pathItems = $currentPath -split ';' | Where-Object { $_ -and $_ -ne '' }

$added = $false
if ($pathItems -notcontains $gitBinPath) {
    $pathItems += $gitBinPath
    $added = $true
}
if ($pathItems -notcontains $gitCmdPath) {
    $pathItems += $gitCmdPath
    $added = $true
}

if ($added) {
    $env:PATH = $pathItems -join ';'
    Write-Host "✓ Git 已添加到 PATH" -ForegroundColor Green
} else {
    Write-Host "✓ Git 已在 PATH 中" -ForegroundColor Green
}

# 测试 Git
Write-Host ""
Write-Host "测试 Git..." -ForegroundColor Yellow
try {
    $version = & git --version 2>&1
    if ($LASTEXITCODE -eq 0) {
        Write-Host $version -ForegroundColor Green
        Write-Host ""
        Write-Host "✓ 成功! 现在可以使用 'git' 命令了" -ForegroundColor Green
        Write-Host ""
        Write-Host "示例:" -ForegroundColor Cyan
        Write-Host "  git init" -ForegroundColor White
        Write-Host "  git --version" -ForegroundColor White
        Write-Host "  git status" -ForegroundColor White
    } else {
        Write-Host "警告: Git 测试失败" -ForegroundColor Yellow
        Write-Host "尝试使用完整路径: .\git-portable\bin\git.exe --version" -ForegroundColor Yellow
    }
} catch {
    Write-Host "错误: 无法运行 Git" -ForegroundColor Red
    Write-Host $_.Exception.Message -ForegroundColor Red
    Write-Host ""
    Write-Host "备用方案: 使用完整路径" -ForegroundColor Yellow
    Write-Host "  .\git-portable\bin\git.exe init" -ForegroundColor White
}

