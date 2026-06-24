#!/usr/bin/env bash
# Download pre-trained checkpoints for HORTOV2 evaluation
# Target layout (relative to this script's directory):
#   checkpoints/hortov2/<network>-LazyTripletLoss_L2/best_model.pth
#
# The file is hosted on a Synology Drive NAS and requires a two-step auth:
#   1. visit the share URL to obtain a session cookie
#   2. call the SynologyDrive Files/download API with the static sharing_token

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CHECKPOINTS_DIR="$SCRIPT_DIR/checkpoints/hortov2"
TMP_DIR="$(mktemp -d)"
ZIP_FILE="$TMP_DIR/trained_on_HORTO3DLM.zip"

echo "[INFO] Downloading trained_on_HORTO3DLM.zip (~587 MB) ..."

python3 - "$ZIP_FILE" <<'PYEOF'
import sys, json, time, requests

OUT     = sys.argv[1]
BASE    = "https://nas-greenbotics.isr.uc.pt"
PERM    = "18mu0TVuBveVvhW9sKGSKS9Yt6OOqskq"
SID     = "YphcgWDv8e11o-cB9e8dlk4SzjSWjxQU-A74AzvB4TA0"
FILE_ID = "id:958273620763782067"
TOKEN   = ("ydcBE6fkbp0LIWc.2DKoTcR6WG3oyd1_7wbqs2Syd89RH5316KQu8ShZFc6PBkRfTQ"
           "UnNpUrJJm20XJlfopxKbQomlOwUmDoEPySzEKxI_edzA1FMFOSIq2eIDZOPg0J2bBQOv"
           "Uv6bPys9Fw.La_ikvSR5cYByabojWTZLqAYBCykP2c3pADmI19WCLz2mHZGI8Fq7NIww"
           "6rIS67oHARwnzfCW0SDA7FL3DuORpjPGRxOPtiXwaUZWe9")
WEBAPI  = f"{BASE}/drive/d/s/{PERM}/webapi/entry.cgi/trained_on_HORTO3DLM.zip"

s = requests.Session()
s.headers.update({"User-Agent": "Mozilla/5.0"})
s.get(f"{BASE}/drive/d/s/{PERM}/{SID}")  # sets the drive-sharing cookie

params = {
    "api":            "SYNO.SynologyDrive.Files",
    "method":         "download",
    "version":        "2",
    "files":          json.dumps([FILE_ID]),
    "force_download": "true",
    "json_error":     "true",
    "c2_offload":     '"allow"',
    "_dc":            str(int(time.time() * 1000)),
    "sharing_token":  f'"{TOKEN}"',
}
r = s.get(WEBAPI, params=params, stream=True)
r.raise_for_status()

total = int(r.headers.get("content-length", 0))
downloaded = 0
with open(OUT, "wb") as f:
    for chunk in r.iter_content(chunk_size=1024 * 1024):
        if chunk:
            f.write(chunk)
            downloaded += len(chunk)
            if total:
                pct = downloaded / total * 100
                mb  = downloaded / 1024 / 1024
                print(f"\r  {pct:5.1f}%  {mb:.0f} MB", end="", flush=True)
print()
print(f"[INFO] Saved to {OUT}")
PYEOF

echo "[INFO] Extracting ZIP..."
unzip -q "$ZIP_FILE" -d "$TMP_DIR/extracted"

# Locate hortov2 folder inside extracted tree
EXTRACTED_HORTOV2=$(find "$TMP_DIR/extracted" -type d -name "hortov2" | head -1)

if [ -n "$EXTRACTED_HORTOV2" ]; then
    echo "[INFO] Copying checkpoints from: $EXTRACTED_HORTOV2"
    mkdir -p "$CHECKPOINTS_DIR"
    cp -r "$EXTRACTED_HORTOV2"/. "$CHECKPOINTS_DIR/"
else
    # Fallback: look for network sub-folders directly
    NETWORKS=("PointNetPGAP-LazyTripletLoss_L2"
              "PointNetVLAD-LazyTripletLoss_L2"
              "SPVSoAP3D-LazyTripletLoss_L2"
              "LOGG3D-LazyTripletLoss_L2"
              "overlap_transformer-LazyTripletLoss_L2")
    FOUND=0
    for NET in "${NETWORKS[@]}"; do
        NET_DIR=$(find "$TMP_DIR/extracted" -type d -name "$NET" | head -1)
        if [ -n "$NET_DIR" ]; then
            echo "[INFO] Copying $NET..."
            cp -r "$NET_DIR" "$CHECKPOINTS_DIR/"
            FOUND=$((FOUND + 1))
        fi
    done
    if [ "$FOUND" -eq 0 ]; then
        echo "[WARNING] Could not place checkpoints automatically. Extracted tree:"
        find "$TMP_DIR/extracted" -maxdepth 3
        echo "[INFO] Please manually copy checkpoint folders into: $CHECKPOINTS_DIR"
        rm -rf "$TMP_DIR"
        exit 1
    fi
fi

rm -rf "$TMP_DIR"

# Verify
echo ""
echo "[INFO] Verifying checkpoint structure..."
NETWORKS=("PointNetPGAP" "PointNetVLAD" "SPVSoAP3D" "LOGG3D" "overlap_transformer")
ALL_OK=true
for NET in "${NETWORKS[@]}"; do
    MODEL="$CHECKPOINTS_DIR/${NET}-LazyTripletLoss_L2/best_model.pth"
    if [ -f "$MODEL" ]; then
        echo "  [OK]  $NET"
    else
        echo "  [MISSING] $MODEL"
        ALL_OK=false
    fi
done

if $ALL_OK; then
    echo ""
    echo "[SUCCESS] All checkpoints are in place. You can now run:"
    echo "  python script_eval_hortov2.py"
else
    echo ""
    echo "[WARNING] Some checkpoints are missing. Check the output above."
    exit 1
fi
