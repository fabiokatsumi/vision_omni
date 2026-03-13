"""Auto-download OmniParser model weights from HuggingFace."""

import os
import shutil


REPO_ID = "microsoft/OmniParser-v2.0"

ICON_DETECT_FILES = [
    "icon_detect/model.pt",
    "icon_detect/train_args.yaml",
    "icon_detect/model.yaml",
]

ICON_CAPTION_FILES = [
    "icon_caption/config.json",
    "icon_caption/generation_config.json",
    "icon_caption/model.safetensors",
]


def ensure_weights(weights_dir: str):
    """Check if weights exist; if not, download from HuggingFace."""
    weights_dir = os.path.abspath(weights_dir)
    os.makedirs(weights_dir, exist_ok=True)

    icon_detect_dir = os.path.join(weights_dir, "icon_detect")
    icon_caption_dir = os.path.join(weights_dir, "icon_caption_florence")

    # Check if all files already exist
    all_exist = True
    for f in ICON_DETECT_FILES:
        path = os.path.join(weights_dir, f)
        if not os.path.exists(path):
            all_exist = False
            break

    if all_exist:
        # Check caption files (in renamed dir)
        for f in ICON_CAPTION_FILES:
            renamed = f.replace("icon_caption/", "icon_caption_florence/")
            path = os.path.join(weights_dir, renamed)
            if not os.path.exists(path):
                all_exist = False
                break

    if all_exist:
        print(f"All weights found in {weights_dir}")
        return

    print(f"Downloading weights from {REPO_ID}...")
    from huggingface_hub import hf_hub_download

    # Download icon detection files
    for f in ICON_DETECT_FILES:
        dest = os.path.join(weights_dir, f)
        if not os.path.exists(dest):
            print(f"  Downloading {f}...")
            hf_hub_download(
                repo_id=REPO_ID,
                filename=f,
                local_dir=weights_dir,
            )

    # Download icon caption files
    for f in ICON_CAPTION_FILES:
        dest_renamed = os.path.join(weights_dir, f.replace("icon_caption/", "icon_caption_florence/"))
        if not os.path.exists(dest_renamed):
            print(f"  Downloading {f}...")
            hf_hub_download(
                repo_id=REPO_ID,
                filename=f,
                local_dir=weights_dir,
            )

    # Rename icon_caption -> icon_caption_florence if needed
    icon_caption_src = os.path.join(weights_dir, "icon_caption")
    if os.path.exists(icon_caption_src) and not os.path.exists(icon_caption_dir):
        print("  Renaming icon_caption -> icon_caption_florence")
        shutil.move(icon_caption_src, icon_caption_dir)

    print("Weights download complete!")
