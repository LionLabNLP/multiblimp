"""Map the Subgender=Masc{1,2,3} values in UD_Polish-LFG to the standard UD 
Animacy feature, which is also used by Polish MPDT, PDB, and PUD.
Values correspond to the standard UD Animacy values, according to UD
documentation (https://universaldependencies.org/pl/feat/SubGender.html):

  SubGender=Masc1 -> Animacy=Hum
  SubGender=Masc2 -> Animacy=Nhum
  SubGender=Masc3 -> Animacy=Inan

Changes: Remap SubGender to Animacy in column 6 (FEATS) of token lines, then
resort alphabetically per UD convention.

XPOS is unchanged, although it contains the same information (m3, m2, m1), but 
this is in line with PDB and PUD.

MDPT uses manim1, manim2 in XPOS (only Hum vs. Nhum distinction in Animcay),
but does not have SubGender in FEATS.


**Previously**
SubGender on tokens 1, 2.

# sent_id = train-1
# text = 100-tysięcznym Grudziądzem rządzi lewica.
# converted_from_file = NKJP1M_NIE_morph_35-p_morph_35.5-s-dis@1.xml
# genre = news
1	100-tysięcznym	100-tysięczny	ADJ	adj:sg:inst:m3:pos	Case=Ins|Degree=Pos|Gender=Masc|Number=Sing|SubGender=Masc3	2	amod	2:amod	_
2	Grudziądzem	Grudziądz	PROPN	subst:sg:inst:m3	Case=Ins|Gender=Masc|Number=Sing|SubGender=Masc3	3	obj	3:obj	_
3	rządzi	rządzić	VERB	fin:sg:ter:imperf	Aspect=Imp|Mood=Ind|Number=Sing|Person=3|Tense=Pres|VerbForm=Fin|Voice=Act	0	root	0:root	_
4	lewica	lewica	NOUN	subst:sg:nom:f	Case=Nom|Gender=Fem|Number=Sing	3	nsubj	3:nsubj	SpaceAfter=No
5	.	.	PUNCT	interp	PunctType=Peri	3	punct	3:punct	_

**After remapping SubGender to Animacy**
Animacy on tokens 1, 2 (Masc3 -> Inan); SubGender removed. XPOS unchanged.

# sent_id = train-1
# text = 100-tysięcznym Grudziądzem rządzi lewica.
# converted_from_file = NKJP1M_NIE_morph_35-p_morph_35.5-s-dis@1.xml
# genre = news
1	100-tysięcznym	100-tysięczny	ADJ	adj:sg:inst:m3:pos	Animacy=Inan|Case=Ins|Degree=Pos|Gender=Masc|Number=Sing	2	amod	2:amod	_
2	Grudziądzem	Grudziądz	PROPN	subst:sg:inst:m3	Animacy=Inan|Case=Ins|Gender=Masc|Number=Sing	3	obj	3:obj	_
3	rządzi	rządzić	VERB	fin:sg:ter:imperf	Aspect=Imp|Mood=Ind|Number=Sing|Person=3|Tense=Pres|VerbForm=Fin|Voice=Act	0	root	0:root	_
4	lewica	lewica	NOUN	subst:sg:nom:f	Case=Nom|Gender=Fem|Number=Sing	3	nsubj	3:nsubj	SpaceAfter=No
5	.	.	PUNCT	interp	PunctType=Peri	3	punct	3:punct	_


Usage:
  python3 fix_polish_subgender_animacy.py --check infile.conllu   # report only
  python3 fix_polish_subgender_animacy.py infile.conllu outfile.conllu  # apply changes and write to outfile.conllu
  

Apply to all three LFG splits:
  for f in train dev test; do
      python3 fix_polish_subgender_animacy.py \
          pl_lfg-ud-$f.conllu pl_lfg-ud-$f.fixed.conllu
  done
"""
import argparse
import sys

SUBGENDER_TO_ANIMACY = {"Masc1": "Hum", "Masc2": "Nhum", "Masc3": "Inan"}


def add_animacy(feats: str) -> tuple[str, bool]:
    """(possibly-modified feats, whether it changed). No-op if there's no
    SubGender=Masc{1,2,3}, or if Animacy is already present -- never
    observed in practice (see module docstring), but don't silently
    overwrite a real value if some future treebank update adds one."""
    if feats in ("_", ""):
        return feats, False
    feats_dict = dict([p.split("=", 1) for p in feats.split("|")])

    animacy = SUBGENDER_TO_ANIMACY.get(feats_dict.get("SubGender"))
    if animacy:
        feats_dict["Animacy"] = animacy
        del feats_dict["SubGender"]
        out_feats = "|".join(f"{k}={v}" for k, v in sorted(feats_dict.items(), key=lambda p: p[0].lower()))
        return out_feats, True
    else:
        return feats, False


def process_file(in_path: str, out_path: str | None) -> dict:
    n_tokens = n_changed = 0
    out_lines = []
    with open(in_path, encoding="utf-8") as f:
        for line in f:
            stripped = line.rstrip("\n")
            cols = stripped.split("\t")
            # Comments (# ...), blank lines, and anything not a 10-column
            # token/multiword-token/empty-node line pass through untouched.
            if len(cols) != 10:
                out_lines.append(line)
                continue
            n_tokens += 1
            new_feats, changed = add_animacy(cols[5])
            if changed:
                cols[5] = new_feats
                n_changed += 1
            out_lines.append("\t".join(cols) + "\n")

    if out_path:
        with open(out_path, "w", encoding="utf-8") as f:
            f.writelines(out_lines)
    return {"tokens": n_tokens, "changed": n_changed}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("in_path")
    parser.add_argument("out_path", nargs="?", default=None,
                         help="Output path. Omit with --check to only report counts.")
    parser.add_argument("--check", action="store_true",
                         help="Report how many tokens would change; write nothing.")
    args = parser.parse_args()

    if not args.check and not args.out_path:
        sys.exit("out_path is required unless --check is given")

    stats = process_file(args.in_path, None if args.check else args.out_path)
    print(f"{args.in_path}: {stats['changed']}/{stats['tokens']} tokens gained Animacy"
          + (" (--check: nothing written)" if args.check else f" -> {args.out_path}"))
