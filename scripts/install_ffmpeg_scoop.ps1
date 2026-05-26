# User-level Scoop installer for ffmpeg
$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot

if (-not (Get-Command scoop -ErrorAction SilentlyContinue)) {
    Write-Host 'Installing Scoop...'
    iwr -useb get.scoop.sh | iex
}

$profileShims = Join-Path $env:USERPROFILE 'scoop\shims'
$env:Path = "$env:Path;$profileShims"

Write-Host 'Installing ffmpeg via Scoop...'
scoop install ffmpeg

$ffmpegPath = Join-Path $profileShims 'ffmpeg.exe'
if (Test-Path $ffmpegPath) {
    & $ffmpegPath -version
} else {
    Write-Error 'ffmpeg executable not found after scoop install.'
    exit 1
}
