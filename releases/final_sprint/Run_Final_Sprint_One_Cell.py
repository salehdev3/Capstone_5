# Single Colab cell. Original Phase 1 submissions and earlier releases are preserved.
from pathlib import Path
import hashlib, json, subprocess, sys, urllib.request, zipfile, os, time, shutil
from google.colab import drive

SOURCE_URL = "https://raw.githubusercontent.com/salehdev3/Capstone_5/1bf35e8896dee153bb306ab3761e058e77145be5/releases/final_sprint/Final_Sprint_Code.zip"
SOURCE_SHA256 = "9315dc335e2d1bdf6f2adfa2ed3aad5c5a3195420dd739c694626268f6be7020"
ROOT = Path("/content/drive/MyDrive/CameraTrap_Phase2")
CACHE = Path("/content/final_sprint_cache")
CODE = Path("/content/final_sprint_code")
print("1. Connecting Drive...", flush=True)
try:
    drive.mount("/content/drive")
except ValueError as error:
    if "must not already contain files" not in str(error):
        raise
    saved = Path("/content") / ("drive_local_backup_" + str(time.time_ns()))
    Path("/content/drive").rename(saved)
    print("Preserved pre-mount local files:", saved, flush=True)
    drive.mount("/content/drive")
manifest = ROOT / "phase1_source/data/cct20_team_v2/manifest.json"
assert (
    manifest.is_file()
), f"Original project metadata is missing at {ROOT}. Mount the Drive account containing your existing CameraTrap_Phase2 project."
print("2. Downloading verified source...", flush=True)
archive = Path("/content/Final_Sprint_Code.zip")
with urllib.request.urlopen(SOURCE_URL, timeout=90) as response:
    archive.write_bytes(response.read())
assert (
    hashlib.sha256(archive.read_bytes()).hexdigest() == SOURCE_SHA256
), "Downloaded source checksum differs"
CODE.mkdir(exist_ok=True)
with zipfile.ZipFile(archive) as package:
    for name in package.namelist():
        target = (CODE / name).resolve()
        assert target.is_relative_to(CODE.resolve()), "Unsafe archive path"
    package.extractall(CODE)
print("3. Checking GPU...", flush=True)
import torch

assert (
    torch.cuda.is_available()
), "Choose Runtime > Change runtime type > GPU, then rerun this same cell. Drive progress is preserved."
print("GPU:", torch.cuda.get_device_name(0), flush=True)
print("4. Installing dependencies...", flush=True)
subprocess.run(
    [
        sys.executable,
        "-m",
        "pip",
        "install",
        "-q",
        "-r",
        str(CODE / "requirements.txt"),
    ],
    check=True,
)
print(
    "5. Running all stages. Progress, logs and resume checkpoints are saved in Drive.",
    flush=True,
)
command = [
    sys.executable,
    "-u",
    str(CODE / "runner.py"),
    "--root",
    str(ROOT),
    "--cache",
    str(CACHE),
]
process = subprocess.Popen(command, start_new_session=True)
try:
    code = process.wait()
    if code:
        raise subprocess.CalledProcessError(code, command)
except KeyboardInterrupt:
    import signal

    os.killpg(process.pid, signal.SIGTERM)
    try:
        process.wait(timeout=15)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
    print(
        "Stopped. Completed stages and epoch checkpoints remain in Drive.", flush=True
    )
    raise
except subprocess.CalledProcessError:
    progress = ROOT / "phase2_results/final_sprint_v1/progress.json"
    if progress.exists():
        print(progress.read_text(), flush=True)
    raise
OUT = ROOT / "phase2_results/final_sprint_v1"
print("6. Optional W&B backup...", flush=True)
key = None
try:
    from google.colab import userdata

    key = userdata.get("WANDB_API_KEY")
except Exception:
    pass
if key:
    try:
        subprocess.run(
            [sys.executable, "-u", str(CODE / "backup.py"), str(ROOT)],
            env=dict(os.environ, WANDB_API_KEY=key),
            check=True,
            timeout=240,
        )
    except (subprocess.TimeoutExpired, subprocess.CalledProcessError) as error:
        print(
            "W&B backup receipt is not confirmed; your local Drive package is already saved.",
            flush=True,
        )
else:
    print(
        "No W&B key available to this notebook; the measured Drive package is saved.",
        flush=True,
    )
print("\nOUTPUTS:", OUT / "submission", flush=True)
print("PACKAGE:", ROOT / "submission_packages/Final_Sprint_Evidence.zip", flush=True)
print(
    "Preset deployment decision:",
    json.loads((OUT / "decision.json").read_text())["performance_decision"],
    flush=True,
)
import pandas as pd
from IPython.display import display

print("Measured criteria:", flush=True)
display(pd.read_csv(OUT / "acceptance.csv"))
print("Above-80% optimization objectives (rate metrics):", flush=True)
display(
    pd.DataFrame(
        json.loads((OUT / "improvement_targets.json").read_text())["measurements"]
    )
)

print(
    "Review the generated report, slide deck, video and failure gallery before submission.",
    flush=True,
)
