"""
Central configuration for FraudMesh.
Edit DATA_DIR to point at your local IEEE-CIS CSVs
(download from https://www.kaggle.com/c/ieee-fraud-detection/data).
"""
import os

# ---- Paths ----
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
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

for d in (DATA_DIR, RESULTS_DIR, MODELS_DIR):
    os.makedirs(d, exist_ok=True)
