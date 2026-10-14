"""UniMorph codes with no UD value, counted over the converted tables (resources/um_tables):
docs/um_values_without_ud.tsv. Run scripts/um2ud.py first."""
import os
import sys
from collections import defaultdict

import pandas as pd

sys.path.append("../src")

from multiblimp.config import REPO_ROOT, RESOURCE_DIR
from multiblimp.languages import get_ud_langs, lang2langcode, skip_langs
from multiblimp.ud_table import SHADOW_PREFIX, UNMAPPED_PREFIX
from multiblimp.um2ud import UM_TABLE_DIR

OUT = os.path.join(REPO_ROOT, "docs", "um_values_without_ud.tsv")


def main():
    code2lang = {lang2langcode(l): l for l in get_ud_langs(RESOURCE_DIR) if l not in skip_langs}
    counts = defaultdict(lambda: defaultdict(int))
    for fname in sorted(os.listdir(UM_TABLE_DIR)):
        if not fname.endswith(".parquet"):
            continue
        code = fname[: -len(".parquet")]
        if code not in code2lang:
            continue
        df = pd.read_parquet(os.path.join(UM_TABLE_DIR, fname))
        for col in df.columns:
            if col in ("lemma", "form", "upos", "frequency") or col.startswith(SHADOW_PREFIX):
                continue
            values = df[col].dropna().astype(str)
            for value, n in values[values.str.startswith(UNMAPPED_PREFIX)].value_counts().items():
                counts[(value[len(UNMAPPED_PREFIX):], col)][code2lang.get(code, code)] += int(n)
    rows = []
    for (tag, col), langs in counts.items():
        per = sorted(langs.items(), key=lambda x: -x[1])
        rows.append((tag, col, sum(langs.values()), len(langs), "; ".join(f"{l} {n}" for l, n in per)))
    rows.sort(key=lambda r: -r[2])
    with open(OUT, "w") as f:
        f.write("tag\tcolumn\trows_in_table\tlanguages\tper_language\n")
        for r in rows:
            f.write("\t".join(map(str, r)) + "\n")
    print(f"{len(rows)} codes -> {OUT}")


if __name__ == "__main__":
    main()
