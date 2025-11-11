# Git 便携版使用说明

Git 便携版已安装到 `git-portable` 目录。

## 快速开始

### 方法 1：在当前 PowerShell 会话中使用（推荐）

在 PowerShell 中运行以下命令：

```powershell
. .\use_git.ps1
```

然后就可以直接使用 `git` 命令了：

```powershell
git init
git --version
```

### 方法 2：手动添加到 PATH

在 PowerShell 中运行：

```powershell
$scriptDir = Get-Location
$gitBin = Join-Path $scriptDir "git-portable\bin"
$gitCmd = Join-Path $scriptDir "git-portable\cmd"
$env:PATH = $env:PATH + ";" + $gitBin + ";" + $gitCmd
git --version
```

### 方法 3：使用完整路径

直接使用完整路径运行 Git：

```powershell
.\git-portable\bin\git.exe init
.\git-portable\cmd\git.exe --version
```

## 永久添加到系统 PATH（可选）

如果希望在所有 PowerShell 会话中都能使用 Git，可以将其添加到用户 PATH：

```powershell
$scriptDir = "D:\研究生文件\比✌\intent-trans"
$gitBin = Join-Path $scriptDir "git-portable\bin"
$gitCmd = Join-Path $scriptDir "git-portable\cmd"
$userPath = [Environment]::GetEnvironmentVariable("Path", "User")
[Environment]::SetEnvironmentVariable("Path", "$userPath;$gitBin;$gitCmd", "User")
```

**注意**：需要重启 PowerShell 才能生效。

## 验证安装

运行以下命令验证 Git 是否可用：

```powershell
git --version
```

应该显示：`git version 2.43.0.windows.1`

## 常见问题

### 问题：命令找不到 git

**解决方案**：运行 `.\use_git.ps1` 将 Git 添加到当前会话的 PATH。

### 问题：每次打开 PowerShell 都需要重新设置

**解决方案**：使用方法 3 将 Git 永久添加到系统 PATH，或者将 `.\use_git.ps1` 添加到 PowerShell 配置文件。

