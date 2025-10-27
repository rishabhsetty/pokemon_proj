# Attacking type rows → defending type columns. Values in {0.0, 0.5, 1.0, 2.0}
# Modern 18-type chart (no GO scaling). Keep keys Title-Case for consistency.
TYPES = [
    "Normal","Fire","Water","Electric","Grass","Ice","Fighting","Poison","Ground",
    "Flying","Psychic","Bug","Rock","Ghost","Dragon","Dark","Steel","Fairy"
]

# Fill the full 18x18 matrix. Here’s a compact way:
# Start with 1.0 everywhere, then overwrite exceptions.
import numpy as np
TYPE_CHART = {atk: {defn: 1.0 for defn in TYPES} for atk in TYPES}

def s(atk, defn, val): TYPE_CHART[atk][defn] = val

# --- Zeroes ---
s("Normal","Ghost",0.0)
s("Fighting","Ghost",0.0)
s("Poison","Steel",0.0)
s("Ground","Flying",0.0)
s("Ghost","Normal",0.0)
s("Electric","Ground",0.0)
s("Dragon","Fairy",0.0)

# --- Halves (0.5) ---
for p in [("Normal","Rock"),("Normal","Steel"),
          ("Fire","Fire"),("Fire","Water"),("Fire","Rock"),("Fire","Dragon"),
          ("Water","Water"),("Water","Grass"),("Water","Dragon"),
          ("Electric","Electric"),("Electric","Grass"),("Electric","Dragon"),
          ("Grass","Fire"),("Grass","Grass"),("Grass","Poison"),("Grass","Flying"),("Grass","Bug"),("Grass","Dragon"),("Grass","Steel"),
          ("Ice","Fire"),("Ice","Water"),("Ice","Ice"),("Ice","Steel"),
          ("Fighting","Poison"),("Fighting","Flying"),("Fighting","Psychic"),("Fighting","Bug"),("Fighting","Fairy"),
          ("Poison","Poison"),("Poison","Ground"),("Poison","Rock"),("Poison","Ghost"),
          ("Ground","Bug"),
          ("Flying","Rock"),("Flying","Steel"),("Flying","Electric"),
          ("Psychic","Psychic"),("Psychic","Steel"),
          ("Bug","Fighting"),("Bug","Flying"),("Bug","Poison"),("Bug","Ghost"),("Bug","Steel"),("Bug","Fire"),("Bug","Fairy"),
          ("Rock","Fighting"),("Rock","Ground"),("Rock","Steel"),
          ("Ghost","Dark"),
          ("Dragon","Steel"),
          ("Dark","Fighting"),("Dark","Dark"),("Dark","Fairy"),
          ("Steel","Steel"),("Steel","Fire"),("Steel","Water"),("Steel","Electric"),
          ("Fairy","Fire"),("Fairy","Poison"),("Fairy","Steel")]:
    s(*p, 0.5)

# --- Doubles (2.0) ---
for p in [("Fire","Grass"),("Fire","Ice"),("Fire","Bug"),("Fire","Steel"),
          ("Water","Fire"),("Water","Ground"),("Water","Rock"),
          ("Electric","Water"),("Electric","Flying"),
          ("Grass","Water"),("Grass","Ground"),("Grass","Rock"),
          ("Ice","Grass"),("Ice","Ground"),("Ice","Flying"),("Ice","Dragon"),
          ("Fighting","Normal"),("Fighting","Rock"),("Fighting","Steel"),("Fighting","Ice"),("Fighting","Dark"),
          ("Poison","Grass"),("Poison","Fairy"),
          ("Ground","Poison"),("Ground","Rock"),("Ground","Steel"),("Ground","Fire"),("Ground","Electric"),
          ("Flying","Grass"),("Flying","Fighting"),("Flying","Bug"),
          ("Psychic","Fighting"),("Psychic","Poison"),
          ("Bug","Grass"),("Bug","Psychic"),("Bug","Dark"),
          ("Rock","Flying"),("Rock","Bug"),("Rock","Fire"),("Rock","Ice"),
          ("Ghost","Psychic"),("Ghost","Ghost"),
          ("Dragon","Dragon"),
          ("Dark","Psychic"),("Dark","Ghost"),
          ("Steel","Rock"),("Steel","Ice"),("Steel","Fairy"),
          ("Fairy","Fighting"),("Fairy","Dragon"),("Fairy","Dark")]:
    s(*p, 2.0)

def type_multiplier(attack_type: str, defend_types: list[str]) -> float:
    """Multiply effectiveness across 1–2 defending types."""
    if attack_type not in TYPE_CHART: return 1.0
    mult = 1.0
    for dt in defend_types:
        if not dt: continue
        mult *= TYPE_CHART[attack_type].get(dt, 1.0)
    return mult
