# Git 便携版使用说明

Git 便携版已安装到 `git-portable` 目录。

## 快速使用

### 方法 1: 在当前 PowerShell 会话中启用 Git

在 PowerShell 中运行以下命令：

```powershell
. .\setup_git_path.ps1
```

然后就可以直接使用 `git` 命令了：

```powershell
git --version
git init
```

### 方法 2: 使用完整路径

直接使用 Git 的完整路径：

```powershell
.\git-portable\bin\git.exe --version
.\git-portable\bin\git.exe init
```

### 方法 3: 永久添加到系统 PATH（推荐）

#### 选项 A: 仅添加到当前用户的 PATH（不需要管理员权限）

在 PowerShell 中运行：

```powershell
$gitBinPath = (Resolve-Path ".\git-portable\bin").Path
$gitCmdPath = (Resolve-Path ".\git-portable\cmd").Path
$userPath = [Environment]::GetEnvironmentVariable("Path", "User")
[Environment]::SetEnvironmentVariable("Path", "$userPath;$gitBinPath;$gitCmdPath", "User")
```

然后**重新打开 PowerShell**，Git 就可以使用了。

#### 选项 B: 添加到系统 PATH（需要管理员权限）

以管理员身份运行 PowerShell，然后执行：

```powershell
$gitBinPath = (Resolve-Path ".\git-portable\bin").Path
$gitCmdPath = (Resolve-Path ".\git-portable\cmd").Path
$systemPath = [Environment]::GetEnvironmentVariable("Path", "Machine")
[Environment]::SetEnvironmentVariable("Path", "$systemPath;$gitBinPath;$gitCmdPath", "Machine")
```

### 方法 4: 自动加载（每次打开 PowerShell 时自动添加）

在 PowerShell 配置文件中添加自动加载脚本。首先检查配置文件是否存在：

```powershell
Test-Path $PROFILE
```

如果返回 `False`，创建配置文件：

```powershell
New-Item -Path $PROFILE -Type File -Force
```

然后编辑配置文件，添加以下内容：

```powershell
notepad $PROFILE
```

在打开的文件中添加：

```powershell
# Auto-load Git from project directory
$gitPath = "D:\研究生文件\比✌\intent-trans\git-portable"
if (Test-Path "$gitPath\bin\git.exe") {
    $env:PATH = "$gitPath\bin;$gitPath\cmd;$env:PATH"
}
```

**注意**: 请将路径替换为您的实际项目路径。

## 验证安装

运行以下命令验证 Git 是否可用：

```powershell
git --version
```

应该显示：`git version 2.43.0.windows.1`

## 常见问题

### Q: 为什么 `git` 命令找不到？

A: Git 已经安装，但还没有添加到 PATH 环境变量中。请按照上面的方法 1 或方法 3 添加 Git 到 PATH。

### Q: 每次打开新的 PowerShell 窗口都需要运行 setup_git_path.ps1 吗？

A: 如果使用方法 1，是的。如果想永久使用，请使用方法 3 或方法 4。

### Q: 如何卸载？

A: 直接删除 `git-portable` 目录即可。如果已添加到系统 PATH，需要手动从环境变量中删除相应路径。

