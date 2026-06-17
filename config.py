# ============================================================
# SAM2 Pipeline Configuration
# ============================================================
# Adjust these paths to match your environment before running.

from pathlib import Path

# --- Project root ---
PROJECT_ROOT = Path(__file__).resolve().parent
ALIGNMENT_PACKAGE = PROJECT_ROOT

# --- Manifest files ---
MANIFEST_VALID706 = ALIGNMENT_PACKAGE / "valid706_manifest_for_alignment.csv"
MANIFEST_HOLDOUT506 = ALIGNMENT_PACKAGE / "final_holdout506_manifest_for_alignment.csv"

# --- Output directory ---
OUTPUT_DIR = PROJECT_ROOT / "outputs"

# --- SAM2 model configuration ---
# Hydra config name (resolved relative to sam2 package, not filesystem path)
SAM2_MODEL_CFG = "configs/sam2.1/sam2.1_hiera_b+.yaml"
# Filesystem path to checkpoint — ADJUST THIS to your environment
SAM2_CHECKPOINT = "./checkpoints/sam2.1_hiera_base_plus.pt"

# --- Dataset path mapping ---
# The manifest uses Linux-style paths.
# Override the base path here to point to your local MicroMat-3K dataset.
# Set to None to use manifest paths as-is.
DATASET_BASE_OVERRIDE = Path("./MicroMat-3K")

# For manifest paths like:
#   /home/lpy/.../datasets/MicroMat3K_hf/MicroMat3K/img/0001.png
# the path fixer extracts /MicroMat3K/... and joins with override.

# --- Methods to run ---
# Each method is a dict: {name, binary_threshold (or None for soft alpha)}
METHODS = [
    {"name": "sam2_bbox_binary", "binary_threshold": 0.0},
    {"name": "sam2_guided", "binary_threshold": None},
    {"name": "sam2_multiscale", "binary_threshold": 0.0, "multiscale": [0.75, 1.0, 1.25]},
    {"name": "sam2_ensemble_rerank", "binary_threshold": 0.0, "ensemble_rerank": True},
    {"name": "sam2_ensemble_guided", "binary_threshold": None, "ensemble_rerank": True},
    {"name": "sam2_multiscale_guided", "binary_threshold": None, "multiscale": [0.75, 1.0, 1.25]},
    {"name": "sam2_ensemble_multiscale", "binary_threshold": 0.0, "ensemble_rerank": True, "multiscale": [0.75, 1.0, 1.25]},
    {"name": "sam2_ensemble_multiscale_guided", "binary_threshold": None, "ensemble_rerank": True, "multiscale": [0.75, 1.0, 1.25]},
]

# --- Inference settings ---
BATCH_SIZE = 1  # SAM2 processes one image at a time
DEVICE = "cuda"  # "cuda" or "cpu"

# --- Metrics settings ---
BOUNDARY_WIDTH = 5  # Width of boundary region for Boundary_SAD
# Boundary is computed via dilation of the GT alpha edge

# --- Validation ---
# Expected number of samples
EXPECTED_N_VALID706 = 706
EXPECTED_N_HOLDOUT506 = 506
