"""
Central configuration for FraudMesh.
Edit DATA_DIR to point at your local IEEE-CIS CSVs
(download from https://www.kaggle.com/c/ieee-fraud-detection/data).

On Kaggle, paths are auto-detected instead (see _kaggle_data_dir below):
competition data is read from wherever it's mounted under /kaggle/input,
and results/models are always written under /kaggle/working, since
/kaggle/input is a read-only mount and this repo itself may be sitting on
it (if uploaded as a Kaggle Dataset rather than cloned into /kaggle/working).
"""
import os


def _on_kaggle() -> bool:
    return os.path.isdir("/kaggle/input") or "KAGGLE_KERNEL_RUN_TYPE" in os.environ


def _kaggle_data_dir():
    """Find whichever /kaggle/input/<dataset>/ folder actually holds the
    IEEE-CIS CSVs — the exact dataset/competition slug isn't guaranteed, so
    we search rather than hard-code a path."""
    base = "/kaggle/input"
    if not os.path.isdir(base):
        return None
    for name in sorted(os.listdir(base)):
        candidate = os.path.join(base, name)
        if os.path.isfile(os.path.join(candidate, "train_transaction.csv")):
            return candidate
    return None


# ---- Paths ----
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
ON_KAGGLE = _on_kaggle()

if ON_KAGGLE:
    DATA_DIR = _kaggle_data_dir() or os.path.join(ROOT_DIR, "data")
    # /kaggle/input is read-only, and ROOT_DIR itself may live there if this
    # repo was uploaded as a Kaggle Dataset rather than git-cloned into
    # /kaggle/working — so writes always go to /kaggle/working regardless of
    # where the code lives.
    RESULTS_DIR = "/kaggle/working/results"
    MODELS_DIR = "/kaggle/working/models"
else:
    DATA_DIR = os.path.join(ROOT_DIR, "data")
    RESULTS_DIR = os.path.join(ROOT_DIR, "results")
    MODELS_DIR = os.path.join(ROOT_DIR, "models")

TRANSACTION_CSV = os.path.join(DATA_DIR, "train_transaction.csv")
IDENTITY_CSV = os.path.join(DATA_DIR, "train_identity.csv")

# ---- Columns used to link transactions into a shared-entity graph ----
# Any two transactions that share a value in one of these columns get an edge.
# card1 = primary card identifier, DeviceInfo = device fingerprint.
#
# addr1 (billing region) and P_emaildomain are deliberately EXCLUDED as graph
# link columns, even though they're natural-sounding "shared entity" fields.
# Both are too coarse: addr1 has only ~300-400 unique values and
# P_emaildomain has ~60 (dominated by gmail.com/yahoo.com/hotmail.com), so
# using either as an identity link merges huge, unrelated swaths of
# transactions into one giant connected component and destroys the local
# structure fraud rings actually have. A specific card or device fingerprint
# is a much stronger identity signal than "uses Gmail" or "lives in region
# 87." Both columns remain available as XGBoost frequency-encoded features
# (see data_prep.py) — just not as graph edges.
ENTITY_LINK_COLUMNS = ["card1", "DeviceInfo"]

# ---- Modeling ----
TARGET_COL = "isFraud"
TIME_COL = "TransactionDT"  # seconds since a reference point, used for time-aware split
RANDOM_SEED = 42
TEST_SIZE = 0.2  # last 20% of time is held out (no shuffling — avoids leakage)

# ---- Recall@FPR thresholds to report ----
FPR_TARGETS = [0.01, 0.05]

# ---- GraphSAGE ----
SAGE_HIDDEN_DIM = 64
SAGE_NUM_LAYERS = 2
SAGE_DROPOUT = 0.3
SAGE_EPOCHS = 30
SAGE_LR = 0.005

for d in (RESULTS_DIR, MODELS_DIR):
    os.makedirs(d, exist_ok=True)
# DATA_DIR may be a read-only /kaggle/input mount — never try to create it;
# only ensure it exists locally, where it's expected to be writable.
if not ON_KAGGLE:
    os.makedirs(DATA_DIR, exist_ok=True)
