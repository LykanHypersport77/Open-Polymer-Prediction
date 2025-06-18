import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors
from pathlib import Path

DATA_DIR = Path("data/raw")
OUT_DIR  = Path("data")
OUT_DIR.mkdir(exist_ok=True, parents=True)

def featurise(smiles: str) -> dict:
    m = Chem.MolFromSmiles(smiles)
    if m is None:
        return {}
    d = {name: fn(m) for name, fn in Descriptors.descList}
    d["NumAtoms"] = m.GetNumAtoms()
    d["MolWt"]    = rdMolDescriptors.CalcExactMolWt(m)
    return d

for split in ["train", "test"]:
    df = pd.read_csv(DATA_DIR / f"{split}.csv")

    # -------- RDKit descriptors --------
    feats = df["SMILES"].apply(featurise).apply(pd.Series)

    # -------- stitch everything together --------
    out = pd.concat([df, feats], axis=1)       # keep id, SMILES, targets **and** new features
    out.to_feather(OUT_DIR / f"{split}_feats.ft")

print("✓ Featurisation complete (targets included)")