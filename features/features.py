from __future__ import annotations
import math
import numpy as np
import pandas as pd
from .type_chart import TYPES, type_multiplier

STAT_COLS = ["HP","Attack","Defense","Sp. Atk","Sp. Def","Speed"]
RENAME_MAPS = {
    # Map your CSV columns → canonical names expected here, adjust once.
    "HP":"HP","Attack":"Attack","Defense":"Defense",
    "Sp. Atk":"Sp. Atk","Sp. Def":"Sp. Def","Speed":"Speed",
    "Type 1":"Type 1","Type 2":"Type 2","Name":"Name","Total":"Total"
}

def normalize_df(df: pd.DataFrame) -> pd.DataFrame:
    m = {src: dst for src, dst in RENAME_MAPS.items() if src in df.columns}
    df = df.rename(columns=m)
    # Keep only the needed columns
    need = ["Name","Type 1","Type 2"] + [c for c in STAT_COLS if c in df.columns]
    if "Total" in df.columns:
        need.append("Total")
    df = df[need].copy()
    # Title-case types
    for tcol in ["Type 1","Type 2"]:
        if tcol in df.columns:
            df[tcol] = df[tcol].fillna("").apply(lambda s: s.strip().title())
    return df

def atk_eff(row):
    # weighted offensive power (adjust weights if you like)
    return 0.6*row["Attack"] + 0.4*row["Sp. Atk"]

def bulk_eff(row):
    return 0.4*row["HP"] + 0.3*row["Defense"] + 0.3*row["Sp. Def"]

def bucket(x, edges=(-200,-100,-50,-20,0,20,50,100,200,1e9)):
    # returns int bucket index 0..len(edges)-1
    # edges define open intervals; simple, stable, C-friendly
    for i, e in enumerate(edges):
        if x <= e:
            return i
    return len(edges)-1

def type_mult_attacker_vs_defender(att_types: list[str], def_types: list[str]) -> float:
    # If dual-typed attacker, take the best single type (simple heuristic)
    best = 1.0
    for at in att_types:
        if not at: continue
        best = max(best, type_multiplier(at, def_types))
    return best

def featurize_pair(a_row: pd.Series, b_row: pd.Series, level_a=50, level_b=50) -> dict:
    # type multipliers (A→B and B→A), map to discrete set for parity
    tm_ab = type_mult_attacker_vs_defender([a_row["Type 1"], a_row.get("Type 2","")],
                                           [b_row["Type 1"], b_row.get("Type 2","")])
    tm_ba = type_mult_attacker_vs_defender([b_row["Type 1"], b_row.get("Type 2","")],
                                           [a_row["Type 1"], a_row.get("Type 2","")])
    # Map {0,0.5,1,2,4} to ints {0,1,2,3,4}
    TM_MAP = {0.0:0, 0.5:1, 1.0:2, 2.0:3, 4.0:4}
    def to_tm_bucket(x):
        # tolerate floating noise: snap to nearest of {0,0.5,1,2,4}
        choices = np.array([0.0,0.5,1.0,2.0,4.0])
        return int(TM_MAP[float(choices[np.argmin(abs(choices-x))])])

    a_atk = atk_eff(a_row); b_atk = atk_eff(b_row)
    a_blk = bulk_eff(a_row); b_blk = bulk_eff(b_row)

    feat = {
        "tm_ab": to_tm_bucket(tm_ab),
        "tm_ba": to_tm_bucket(tm_ba),

        
        "d_hp":  bucket(int(a_row["HP"]      - b_row["HP"])),
        "d_atk": bucket(int(a_row["Attack"]  - b_row["Attack"])),
        "d_def": bucket(int(a_row["Defense"] - b_row["Defense"])),
        "d_spa": bucket(int(a_row["Sp. Atk"] - b_row["Sp. Atk"])),
        "d_spd": bucket(int(a_row["Sp. Def"] - b_row["Sp. Def"])),
        "d_spe": bucket(int(a_row["Speed"]   - b_row["Speed"])),

        # total base stat diff if available
        "d_total": bucket(int(a_row.get("Total", a_row[STAT_COLS].sum()) -
                              int(b_row.get("Total", b_row[STAT_COLS].sum())))),

        # initiative flag
        "a_faster": int(a_row["Speed"] > b_row["Speed"]),

        # levels
        "d_level": bucket(int(level_a - level_b), edges=(-50,-20,-10,-5,-1,0,1,5,10,20,50,1e9)),
    }
    return feat

def label_heuristic(a_row, b_row, level_a=50, level_b=50) -> int:
    # Deterministic “sim score” for Option A labels
    tm_ab = {0:0.0,1:0.5,2:1.0,3:2.0,4:4.0}[featurize_pair(a_row,b_row,level_a,level_b)["tm_ab"]]
    tm_ba = {0:0.0,1:0.5,2:1.0,3:2.0,4:4.0}[featurize_pair(b_row,a_row,level_b,level_a)["tm_ab"]]

    a = atk_eff(a_row) * tm_ab * (level_a/50.0)
    b = atk_eff(b_row) * tm_ba * (level_b/50.0)
    a /= max(1.0, bulk_eff(b_row))
    b /= max(1.0, bulk_eff(a_row))
    # small initiative bump
    if a_row["Speed"] > b_row["Speed"]: a *= 1.05
    if b_row["Speed"] > a_row["Speed"]: b *= 1.05
    return int(a > b)
