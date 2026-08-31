import config
from src.data_prep import (
    engineer_features,
    get_feature_columns,
    make_synthetic_dataset,
    time_aware_split,
)


def test_make_synthetic_dataset_fraud_rate_close_to_target():
    df = make_synthetic_dataset(n_rows=5000, fraud_rate=0.035, seed=0)
    assert abs(df["isFraud"].mean() - 0.035) < 0.01


def test_engineer_features_adds_expected_columns():
    df = make_synthetic_dataset(n_rows=200, seed=0)
    out = engineer_features(df)
    for col in ("tx_hour", "tx_day", "TransactionAmt_log", "TransactionAmt_decimal",
                "card1_freq", "DeviceInfo_freq", "P_emaildomain_freq", "addr1_freq"):
        assert col in out.columns


def test_get_feature_columns_excludes_target_and_link_columns(synthetic_df):
    feature_cols = get_feature_columns(synthetic_df)
    assert config.TARGET_COL not in feature_cols
    assert "TransactionID" not in feature_cols
    assert config.TIME_COL not in feature_cols
    for link_col in config.ENTITY_LINK_COLUMNS:
        assert link_col not in feature_cols
    # Frequency-encoded versions of link columns should still be present.
    assert "card1_freq" in feature_cols


def test_time_aware_split_no_time_overlap(synthetic_df):
    train, test = time_aware_split(synthetic_df, test_size=0.2)
    assert len(train) + len(test) == len(synthetic_df)
    assert train[config.TIME_COL].max() <= test[config.TIME_COL].min()
