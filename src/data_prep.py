"""
Load IEEE-CIS transaction data, engineer baseline tabular features, and
produce a time-aware train/test split.

Time-aware split matters: shuffling transactions randomly leaks future
patterns into training and inflates every metric. We sort by TransactionDT
and hold out the last TEST_SIZE fraction as test, exactly as the model would
see data in production.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


def load_raw(sample_frac: float | None = None) -> pd.DataFrame:
    """Load and merge transaction + identity tables.

    sample_frac: if set, randomly subsamples rows AFTER a fraud-preserving
    stratified draw — useful for fast local iteration before a full run.
    """
    if not os.path.exists(config.TRANSACTION_CSV):
        raise FileNotFoundError(
            f"Expected transaction data at {config.TRANSACTION_CSV}. "
            "Download train_transaction.csv (+ train_identity.csv) from "
            "https://www.kaggle.com/c/ieee-fraud-detection/data and place "
            "them in the data/ directory."
        )

    tx = pd.read_csv(config.TRANSACTION_CSV)
    if os.path.exists(config.IDENTITY_CSV):
        ident = pd.read_csv(config.IDENTITY_CSV)
        df = tx.merge(ident, on="TransactionID", how="left")
    else:
        df = tx

    if sample_frac is not None:
        fraud = df[df[config.TARGET_COL] == 1]
        legit = df[df[config.TARGET_COL] == 0].sample(
            frac=sample_frac, random_state=config.RANDOM_SEED
        )
        df = pd.concat([fraud, legit]).sort_values(config.TIME_COL)

    return df.reset_index(drop=True)


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Baseline tabular feature engineering shared by every model in this repo.

    Kept intentionally simple and leakage-safe: every aggregate here is a
    property of the row itself or the raw categorical columns, not a
    target-derived statistic.
    """
    df = df.copy()

    # Time-derived features
    df["tx_hour"] = (df[config.TIME_COL] // 3600) % 24
    df["tx_day"] = df[config.TIME_COL] // (3600 * 24)

    # Amount transforms (fraud amounts are often round numbers or outliers)
    df["TransactionAmt_log"] = np.log1p(df["TransactionAmt"])
    df["TransactionAmt_decimal"] = (
        df["TransactionAmt"] - df["TransactionAmt"].astype(int)
    )

    # Frequency encoding for high-cardinality categoricals — how common is
    # this card / device / email domain in the dataset overall. Computed on
    # the full column, not the target, so it's leakage-safe.
    for col in ["card1", "DeviceInfo", "P_emaildomain", "addr1"]:
        if col in df.columns:
            freq = df[col].value_counts(dropna=False)
            df[f"{col}_freq"] = df[col].map(freq).fillna(0)

    return df


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    """Numeric feature columns for XGBoost, excluding IDs, target, and raw
    high-cardinality categoricals (their frequency-encoded versions are used
    instead)."""
    exclude = {
        config.TARGET_COL,
        "TransactionID",
        config.TIME_COL,
    }
    exclude.update(config.ENTITY_LINK_COLUMNS)  # raw IDs excluded, *_freq kept

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    return [c for c in numeric_cols if c not in exclude]


def time_aware_split(df: pd.DataFrame, test_size: float = config.TEST_SIZE):
    """Sort by transaction time and split so test is strictly later than train."""
    df_sorted = df.sort_values(config.TIME_COL).reset_index(drop=True)
    cutoff = int(len(df_sorted) * (1 - test_size))
    train = df_sorted.iloc[:cutoff].reset_index(drop=True)
    test = df_sorted.iloc[cutoff:].reset_index(drop=True)
    return train, test


def make_synthetic_dataset(n_rows: int = 20000, fraud_rate: float = 0.035,
                            seed: int = config.RANDOM_SEED) -> pd.DataFrame:
    """Generate a synthetic IEEE-CIS-shaped dataset for local testing when the
    real Kaggle data isn't available yet. Mirrors the key columns and injects
    fraud rings (shared card/device clusters with elevated fraud rate) so the
    graph-based models have real signal to find.
    """
    rng = np.random.default_rng(seed)
    n_fraud = int(n_rows * fraud_rate)
    n_legit = n_rows - n_fraud

    # High cardinality for legitimate traffic — most legit transactions use a
    # near-unique card/device, mirroring real payment data where collisions
    # are rare. This matters: if legit cards/devices were also low-cardinality,
    # everything would merge into one giant connected component and the graph
    # would carry no signal (see graph_builder.py's max_edges_per_entity guard).
    n_cards = n_rows * 3
    n_devices = n_rows * 2
    n_domains = 15
    n_addrs = 50

    # Fraud rings: a SMALL set of cards/devices reused heavily among fraud
    # rows only — this is the structural pattern (card testing / device
    # reuse) the graph is meant to expose.
    n_rings = max(3, n_fraud // 40)
    ring_cards = rng.integers(1000, 1000 + n_cards, size=n_rings)
    ring_devices = rng.integers(1, n_devices, size=n_rings)

    def gen_block(n, is_fraud):
        if is_fraud and len(ring_cards) > 0:
            # ~70% of fraud rows come from a small reused ring (detectable
            # structure); the rest are "lone wolf" fraud with unique
            # card/device, same as legit traffic, so the graph baseline can't
            # catch everything — a realistic and honest limitation.
            n_ring = int(n * 0.7)
            n_lone = n - n_ring
            ring_card1 = rng.choice(ring_cards, size=n_ring)
            ring_device = rng.choice(ring_devices, size=n_ring).astype(str)
            lone_card1 = rng.integers(1000, 1000 + n_cards, size=n_lone)
            lone_device = rng.integers(1, n_devices, size=n_lone).astype(str)
            card1 = np.concatenate([ring_card1, lone_card1])
            device = np.concatenate([ring_device, lone_device])
            amt = rng.exponential(scale=250, size=n) + 5  # fraud skews higher amt
        else:
            card1 = rng.integers(1000, 1000 + n_cards, size=n)
            device = rng.integers(1, n_devices, size=n).astype(str)
            amt = rng.exponential(scale=80, size=n) + 1

        return pd.DataFrame({
            "TransactionAmt": amt,
            "card1": card1,
            # np.char.add, not "device_" + device: numpy's .astype(str) on
            # an int64 array always allocates a fixed-width '<U21' dtype
            # (room for any 64-bit int), regardless of the values actually
            # present, while a bare python-str literal promotes to a
            # narrower '<U7'. Plain "+" between mismatched string itemsizes
            # raises a UFuncNoLoopError on some numpy versions (reproduced
            # on 1.26) and silently works on others (2.x) — np.char.add
            # handles the promotion correctly on both.
            "DeviceInfo": np.char.add("device_", device),
            "P_emaildomain": rng.choice(
                [f"domain{i}.com" for i in range(n_domains)], size=n
            ),
            "addr1": rng.integers(100, 100 + n_addrs, size=n),
            "isFraud": int(is_fraud),
        })

    fraud_block = gen_block(n_fraud, True)
    legit_block = gen_block(n_legit, False)
    df = pd.concat([fraud_block, legit_block], ignore_index=True)

    # Assign timestamps independently per row (not pre-sorted by block) so
    # fraud is spread uniformly across the time range, matching reality.
    # Sorting happens later, at split time, in time_aware_split().
    df["TransactionDT"] = rng.integers(0, 3600 * 24 * 30, size=len(df))
    df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    df["TransactionID"] = np.arange(len(df))
    return df


if __name__ == "__main__":
    # Local smoke test using synthetic data
    df = make_synthetic_dataset(n_rows=5000)
    df = engineer_features(df)
    train, test = time_aware_split(df)
    feats = get_feature_columns(df)
    print(f"rows={len(df)} fraud_rate={df['isFraud'].mean():.4f}")
    print(f"train={len(train)} test={len(test)} n_features={len(feats)}")
    print("feature columns:", feats)
