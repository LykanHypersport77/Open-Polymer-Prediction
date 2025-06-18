import pandas as pd, numpy as np, hashlib, os
df = pd.read_csv("data/raw/train.csv")
# Group by repeat unit so similar SMILES stay together
def scaffold_hash(smiles):
    return int(hashlib.md5(smiles.encode()).hexdigest(), 16) % 10
df["fold"] = df["SMILES"].apply(scaffold_hash)
train_idx   = df[df.fold < 8].index   # 80 %
valid_idx   = df[df.fold >= 8].index  # 20 %
np.save("data/splits/train_idx.npy", train_idx)
np.save("data/splits/valid_idx.npy", valid_idx)
print(df.fold.value_counts())