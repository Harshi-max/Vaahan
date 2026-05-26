# Audio Recording Guide — ASR Shootout Dataset

## Naming Convention

Use this format for your recordings:

```
<locality>_<condition>_<number>.<ext>
```

Examples:
- `koramangala_traffic_01.wav` - Recording 1 from Koramangala in traffic
- `whitefield_phonecall_02.m4a` - Recording 2 from Whitefield during phone call
- `bengaluru_quiet_03.mp3` - Recording 3 from Bengaluru in quiet setting
- `indiranagar_whispered_01.wav` - Recording 1 from Indiranagar, whispered

**Components:**
- **`<locality>`**: Location name (e.g., koramangala, whitefield, indiranagar, bengaluru)
- **`<condition>`**: One of: `phonecall`, `quiet`, `traffic`, `rushed`, `whispered`
- **`<number>`**: Zero-padded sequence (01, 02, 03, ...)
- **`<ext>`**: File extension (`.wav`, `.mp3`, `.m4a`)

## Required Metadata (ground_truth.csv)

The metadata CSV is automatically generated from your audio files:

```bash
python src/utils/generate_metadata.py
```

This creates `data/metadata/ground_truth.csv` with columns:

| Column | Description |
|--------|-------------|
| filename | Audio filename (auto-detected) |
| condition | Folder name: `quiet`, `traffic`, `rushed`, `whispered`, `phonecall` |
| language | Language tag, default: `hinglish` |
| locality_name | Extracted from filename (e.g., "koramangala" from "koramangala_traffic_01.wav") |
| reference_transcript | **You must fill this in manually** with the full spoken text |

**After generation:**
1. Open `data/metadata/ground_truth.csv`
2. Fill in `reference_transcript` for each file with what was actually spoken
3. Save and re-run the pipeline

### Auto-generation Options

```bash
# Default: full validation
python src/utils/generate_metadata.py

# Skip corruption checks (faster)
python src/utils/generate_metadata.py --skip-corruption-check

# Skip silence detection
python src/utils/generate_metadata.py --skip-silence-check

# Use custom output path
python src/utils/generate_metadata.py --output-csv custom_metadata.csv

# Verbose logging
python src/utils/generate_metadata.py --verbose
```

## Recording Conditions

| Condition | How to Record | Best Practices |
|-----------|---------------|-----------------|
| `quiet` | Indoor, minimal background noise | Record in a silent room; no AC or traffic noise |
| `traffic` | Near road / open window with traffic | Record at a busy intersection; background noise required |
| `rushed` | Speak quickly, natural fast delivery | Simulate urgency; speak faster than normal |
| `whispered` | Low volume, as if not wanting others to hear | Whisper or speak very softly; still intelligible |
| `phonecall` | Record via phone call or simulate phone audio | Call a friend; use speaker phone or apply band-limit |

## Mobile Recording (Phone)

### iOS (iPhone)

1. Open **Voice Memos** app
2. Tap the **red record button**
3. Speak naturally for 30–60 seconds (include locality name)
4. Tap **Done** to save
5. Long-press the file > **Share** > **Save to Files** or **Cloud Drive**
6. Name the file using convention: `locality_condition_number.m4a`

### Android

1. Open **Google Recorder** or your phone's voice recorder app
2. Tap **Start Recording** / red record button
3. Speak naturally for 30–60 seconds (include locality name)
4. Tap **Stop** to save
5. Select the file > **Share** > **Google Drive** or **Email**
6. Name the file using convention: `locality_condition_number.wav` or `.mp3`

### Recording Tips

- **Speak naturally** using Hinglish (mix of Hindi and English)
- **Include the locality name** somewhere in your speech
- **Speak clearly** for ~30–60 seconds
- **Use phone's default microphone** (no external mic needed)
- **Match the condition:** for "traffic", record with actual road noise
- **Minimize clipping:** speak at normal volume, not too loud

## Transferring to Your Computer

### Option 1: Cloud Storage (Easiest)

1. Upload to **Google Drive**, **OneDrive**, or **Dropbox**
2. Create folder `ASR-Recordings` with subfolders: `phonecall`, `quiet`, `traffic`, `rushed`, `whispered`
3. Upload files to matching folders
4. Download on computer: `gsutil -m cp -r gs://your-bucket/ASR-Recordings ~/Downloads/`
5. Move to `data/raw/`: `cp -r ~/Downloads/ASR-Recordings/* data/raw/`

### Option 2: USB Cable

**iOS:**
- Connect iPhone to Mac via USB
- Open **Finder** > **Your iPhone** > **Files** > **Voice Memos**
- Drag-and-drop `.m4a` files to your computer

**Android:**
- Connect phone via USB
- Enable **File Transfer** mode
- Navigate to **Internal Storage** > **Recordings**
- Drag-and-drop files to your computer

### Option 3: AirDrop or Email

- **AirDrop (iOS):** Select files > Share > AirDrop > your Mac
- **Email:** Attach recordings to self or team email
- Download locally and move to `data/raw/`

## Technical Specs

- **Format:** `.wav`, `.mp3`, `.m4a`, or `.mp4`
- **Sample rate:** Any (auto-resampled to 16 kHz by pipeline)
- **Channels:** Mono or stereo (converted to mono automatically)
- **Duration:** 10–120 seconds per clip
- **Bitrate:** ≥128 kbps for MP3/M4A
- **Content:** Hinglish/English speech; include locality name naturally

> Note: files named like `phone1.mp4`, `phone2.mp4`, or `WhatsApp Audio ... .mp4` are treated as `phonecall` recordings.

## Audio Quality Validation

The metadata generator automatically checks:

- ✅ **File format:** Supports `.wav`, `.mp3`, `.m4a`
- ✅ **Duration:** Must be 0.5–120 seconds (configurable)
- ✅ **Corruption:** Attempts to read audio; warns if unreadable
- ✅ **Silence detection:** Warns if >80% of audio is silence
- ✅ **Duplicates:** Alerts if same filename exists
- ⚠️ **Missing transcripts:** Warns if reference_transcript is empty

### Directory Structure

After recording and uploading, organize your files:

```
data/raw/
├── phonecall/
│   ├── koramangala_phonecall_01.wav
│   ├── whitefield_phonecall_02.m4a
│   └── ...
├── quiet/
│   ├── koramangala_quiet_01.wav
│   └── ...
├── traffic/
│   ├── whitefield_traffic_01.mp3
│   └── ...
├── rushed/
│   └── ...
└── whispered/
    └── ...
```

## Workflow: From Recording to Benchmark

1. **Record** on your phone using Voice Memos or a recording app
2. **Transfer** files to `data/raw/<condition>/` folders
3. **Generate metadata:**
   ```bash
   python src/utils/generate_metadata.py
   ```
4. **Edit metadata:** Open `data/metadata/ground_truth.csv` and fill in `reference_transcript`
5. **Validate dataset:**
   ```bash
   python scripts/validate_dataset.py
   ```
6. **Run benchmark:**
   ```bash
   bash run.sh
   # or
   .\run.ps1  # on Windows
   ```
7. **View results:**
   ```bash
   streamlit run dashboard/app.py
   ```

## Sample Transcripts (Hinglish)

Use these as ideas for what to record:

**Delivery/Logistics Context:**
- "Mera delivery address hai Koramangala 5th block, Bangalore, landmark CCD ke paas."
- "Customer bol raha hai Whitefield metro ke paas pickup karo."
- "Indiranagar mein apartment number 302 pe delivery hai."

**General Hinglish:**
- "Namaste, main Bangalore mein rehta hoon, Koramangala area mein."
- "Aaj weather kaisa hai? Bahut traffic hai roads pe."
- "Mujhe ek coffee lani hai nearest Cafe Coffee Day se."

**Phone-realistic:**
- "Hello? Haan, main yahin hoon. Tum kahan ho?"
- "Arre, signal bahut weak hai idhar. Suno tum?"

**Rushed (fast/urgent):**
- "Jaldi jaldi, server down ho gaya, urgent fix karo!"
- "Arre, delivery boy abhi aa gaya, door kholna!"

**Whispered (soft):**
- "Shhh, quietly listen, location coordinates check karo..."
- "Whisper mein bol raha hoon, coordinates match karna..."

## Troubleshooting

**Q: My file won't transfer via USB**
- A: Ensure phone is set to "File Transfer" mode (not "Charge Only")
- Try a different USB cable or port

**Q: Metadata generator reports "corrupted audio"**
- A: File may be damaged; try re-recording
- Or convert format: `ffmpeg -i broken.mp3 -c:a pcm_s16le output.wav`

**Q: How do I add transcripts to the CSV?**
- A: Open `data/metadata/ground_truth.csv` in Excel/Google Sheets/text editor
- For each row, fill `reference_transcript` with what was actually spoken
- Save and re-validate

**Q: Can I use a different naming format?**
- A: Naming is flexible; the generator extracts locality from first part before `_`
- Stick to `<locality>_<condition>_<number>.<ext>` for consistency

**Q: How many recordings do I need?**
- A: Start with 5–10 per condition (25–50 total)
- More is better for statistical significance

## Quality Checklist

Before submitting:

- [ ] Filenames follow `<locality>_<condition>_<number>.<ext>`
- [ ] Files are in correct condition folder under `data/raw/`
- [ ] Audio duration is 10–120 seconds
- [ ] Format is `.wav`, `.mp3`, or `.m4a`
- [ ] Speech is clear and understandable (SNR > 10dB)
- [ ] No extreme clipping or distortion
- [ ] Background noise matches the condition
- [ ] Transcripts filled in CSV

## Next Steps

1. Run `python src/utils/generate_metadata.py` to validate
2. Fill in `reference_transcript` in `ground_truth.csv`
3. Run `python scripts/validate_dataset.py` for final checks
4. Execute `bash run.sh` or `.\run.ps1` to benchmark
5. View results in `streamlit run dashboard/app.py`

---

**Questions?** Refer to [README.md](../README.md) or check the [ASR models documentation](../README.md#supported-models).
