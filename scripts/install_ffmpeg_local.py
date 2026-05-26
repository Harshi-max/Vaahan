from pathlib import Path
from urllib.request import urlopen
from zipfile import ZipFile

TARGET = Path("tools/ffmpeg")
TARGET.mkdir(parents=True, exist_ok=True)
ZIP_PATH = TARGET / "ffmpeg.zip"
URL = "https://www.gyan.dev/ffmpeg/builds/ffmpeg-release-essentials.zip"

print(f"Downloading {URL}...")
with urlopen(URL) as response, open(ZIP_PATH, "wb") as out_file:
    out_file.write(response.read())
print(f"Downloaded {ZIP_PATH} ({ZIP_PATH.stat().st_size} bytes)")

with ZipFile(ZIP_PATH, "r") as z:
    z.extractall(TARGET)
print(f"Extracted to {TARGET}")

ffmpeg_exe = next(TARGET.rglob("ffmpeg.exe"), None)
if ffmpeg_exe is None:
    raise SystemExit("ffmpeg.exe not found after extraction")

print(f"ffmpeg executable located at: {ffmpeg_exe}")
