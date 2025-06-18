import numpy as np, pandas as pd, lightgbm as lgb
from pathlib import Path
from sklearn.metrics import mean_absolute_error

FEAT_DIR = Path("data")
TRAIN_F  = FEAT_DIR / "train_feats.ft"
TEST_F   = FEAT_DIR / "test_feats.ft"

train_df = pd.read_feather(TRAIN_F)
test_df  = pd.read_feather(TEST_F)

y_cols = ["Density", "Tc", "Tg", "Rg", "FFV"]
X_train = train_df.drop(columns=y_cols + ["SMILES"], errors="ignore")
X_test  = test_df.drop(columns=y_cols + ["SMILES"], errors="ignore")

train_idx = np.load("data/splits/train_idx.npy")
valid_idx = np.load("data/splits/valid_idx.npy")

preds = {}
for target in y_cols:
    y = train_df[target]
    mask_train = y.notna() & np.isin(train_df.index, train_idx)
    mask_valid = y.notna() & np.isin(train_df.index, valid_idx)

    dtrain = lgb.Dataset(X_train.loc[mask_train], y[mask_train])
    dvalid = lgb.Dataset(X_train.loc[mask_valid], y[mask_valid])

    # -------- LightGBM 4.x call --------
    model = lgb.train(
        params=dict(
            objective="regression",
            metric="mae",
            learning_rate=0.03,
            num_leaves=127,
            feature_fraction=0.8,
            bagging_fraction=0.8,
            bagging_freq=1,
            verbose=-1,
        ),
        train_set=dtrain,
        num_boost_round=4000,
        valid_sets=[dvalid],
        # new way: callback objects
        callbacks=[
            lgb.early_stopping(200),   # stop if no valid improvement for 200 rounds
            lgb.log_evaluation(200),   # print every 200 rounds
        ],
    )

    valid_mae = mean_absolute_error(
        y[mask_valid], model.predict(X_train.loc[mask_valid])
    )
    print(f"{target:8} MAE(valid) = {valid_mae:.4f}")

    preds[target] = model.predict(X_test)

sub = pd.DataFrame({"id": test_df["id"]})
for t in y_cols:
    sub[t] = preds[t]
sub.to_csv("submission.csv", index=False)
print("✓ submission.csv written")