# Helper to install ffmpeg on Windows (tries Chocolatey or Scoop)
param()

Write-Host "Attempting to install ffmpeg (requires admin for Chocolatey)"

function Install-Using-Choco {
    choco install ffmpeg -y
}

function Install-Using-Scoop {
    scoop install ffmpeg
}

if (Get-Command choco -ErrorAction SilentlyContinue) {
    Write-Host "Chocolatey found — installing ffmpeg..."
    try {
        Install-Using-Choco
        Write-Host "ffmpeg installed via Chocolatey."
        exit 0
    } catch {
        Write-Warning "Chocolatey install failed: $_"
    }
}

if (Get-Command scoop -ErrorAction SilentlyContinue) {
    Write-Host "Scoop found — installing ffmpeg..."
    try {
        Install-Using-Scoop
        Write-Host "ffmpeg installed via Scoop."
        exit 0
    } catch {
        Write-Warning "Scoop install failed: $_"
    }
}

Write-Host "Could not auto-install ffmpeg. Please manually install from https://ffmpeg.org/download.html"
Write-Host "Or use Chocolatey: 'choco install ffmpeg -y' (run PowerShell as Administrator)"
Write-Host "Or use Scoop: 'scoop install ffmpeg'"
Write-Host "After installation, ensure ffmpeg.exe is on your PATH and restart your shell."
exit 1
