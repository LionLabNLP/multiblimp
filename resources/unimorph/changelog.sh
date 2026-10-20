#!/usr/bin/env bash
# One-time, in-place edits to the downloaded data; skipped once the marker exists.
set -eu
cd "$(dirname "$0")"
MARKER=".changelog_applied"
if [ -e "$MARKER" ]; then
  echo "changelog already applied ($MARKER), skipping"
  exit 0
fi

# Italian
echo -e "essere\tè\tV|IND;PRS;3;SG\tessere|X" >> ita/ita.segmentations

# French
grep -vE "^avoyr\t" fra/fra > fra/fra2; mv fra/fra2 fra/fra
grep -vE "^aveir\t" fra/fra > fra/fra2; mv fra/fra2 fra/fra
grep -vE "^savir\t" fra/fra > fra/fra2; mv fra/fra2 fra/fra
grep -vE "^saveir\t" fra/fra > fra/fra2; mv fra/fra2 fra/fra
grep -vE "^reçoivre\t" fra/fra > fra/fra2; mv fra/fra2 fra/fra
grep -vE "^estre\t" fra/fra > fra/fra2; mv fra/fra2 fra/fra
grep -vE "^estre\t" fra/fra.segmentations > fra/fra2; mv fra/fra2 fra/fra.segmentations
grep -vE "^mectre\t" fra/fra > fra/fra2; mv fra/fra2 fra/fra

echo -e "être\test\tV|IND;PRS;3;SG\têtre|X" >> fra/fra.segmentations
echo -e "être\tsommes\tV|IND;PRS;1;PL\têtre|X" >> fra/fra.segmentations
echo -e "être\têtes\tV|IND;PRS;2;PL\têtre|X" >> fra/fra.segmentations
echo -e "être\tsont\tV|IND;PRS;3;PL\têtre|X" >> fra/fra.segmentations

echo -e "mettre\tmet\tV|IND;PRS;3;SG\têtre|X" >> fra/fra.segmentations

# English
sed -nE "s/([a-z]+)\t[a-z]+\tV;PRS;3;SG/\1\t\1\tV;PRS;3;PL/p" eng/eng >> eng/eng
sed "s/V;PRS;3/V;PRS;3;IND/" eng/eng > eng/eng2; mv eng/eng2 eng/eng
echo -e "be\tam\tV;IND;PRS;1;SG" >> eng/eng
echo -e "be\tare\tV;IND;PRS;2;SG" >> eng/eng
echo -e "be\tis\tV;IND;PRS;3;SG" >> eng/eng
echo -e "be\tare\tV;IND;PRS;1;PL" >> eng/eng
echo -e "be\tare\tV;IND;PRS;2;PL" >> eng/eng
echo -e "be\tare\tV;IND;PRS;3;PL" >> eng/eng

# Finnish
cp fin/fin.1 fin/fin

# Azerbaijani
grep -vE "(dövri|elektron|uçan|xarici)" aze/aze > aze/aze2; mv aze/aze2 aze/aze

# Uzbek
cat uzb/uzb_verbs >> uzb/uzb

# Livvi
cp olo/olo-new-written-livvic olo/olo
tail -n +2 olo/olo > olo/olo2; mv olo/olo2 olo/olo

# Karelian
cat krl/krl-new-written-karelian >> krl/krl
grep -v "#" krl/krl > krl/krl2; mv krl/krl2 krl/krl

# Veps
cp vep/vep-new-written-veps vep/vep

# Faroese
sed -E "s/\t(.)\.(.+)\.(.+)$/\t\1;\2;\3/" fao/fao > fao/fao2; mv fao/fao2 fao/fao
sed -E "s/\t(.)\.(.+)$/\t\1;\2/" fao/fao > fao/fao2; mv fao/fao2 fao/fao

# Korean
grep -vE "^[^\t]+\t[^\t]+\t$" kor/kor > kor/kor2; mv kor/kor2 kor/kor
grep -vE "Formal" kor/kor > kor/kor2; mv kor/kor2 kor/kor
grep -vE "Informal" kor/kor > kor/kor2; mv kor/kor2 kor/kor
sed "s/CV;/CVB;/" kor/kor > kor/kor2; mv kor/kor2 kor/kor

# Basque
sed "s/PRES/PRS/" eus/eus > eus/eus2; mv eus/eus2 eus/eus
sed "s/PAST/PST/" eus/eus > eus/eus2; mv eus/eus2 eus/eus

# Sorbian
sed "s/:/;/" hsb/hsb > hsb/hsb2; mv hsb/hsb2 hsb/hsb
sed "s/:3:/;3;/" hsb/hsb > hsb/hsb2; mv hsb/hsb2 hsb/hsb

# Slovak
unxz slk/slk.xz
awk -F'\t' -v OFS='\t' '{ if ($4 != "") $3 = $3 ";" $4; print $1, $2, $3 }' slk/slk > slk/slk2; mv slk/slk2 slk/slk

# Kazakh
cat kaz/kaz.sm >> kaz/kaz

# Sanskrit
sed "s/PRES/PRS/" san/san > san/san2; mv san/san2 san/san
sed "s/{//g; s/}//g" san/san > san/san2; mv san/san2 san/san

# Dutch
echo -e "willen\twil\tV;IND;PRS;2;SG" >> nld/nld
echo -e "zullen\tzal\tV;IND;PRS;2;SG" >> nld/nld

# German
grep -v ADP deu/deu.segmentations > deu/deu2.segmentations; mv deu/deu2.segmentations deu/deu.segmentations

# Kurmanji
sed "s/1;2;3;/1,2,3;/" kmr/kmr > kmr/kmr2; mv kmr/kmr2 kmr/kmr

# Yiddish
sed "s/PRES/PRS/" yid/yid > yid/yid2; mv yid/yid2 yid/yid

# Irish
sed "s/MASV/MASC/" gle/gle > gle/gle2; mv gle/gle2 gle/gle

# Chuckchi
sed "s/ARBAB1S/ARGAB1S/" ckt/ckt > ckt/ckt2; mv ckt/ckt2 ckt/ckt
sed "s/ARBAB3S/ARGAB3S/" ckt/ckt > ckt/ckt2; mv ckt/ckt2 ckt/ckt

# Afrikaans
sed "s/SUP/SPRL/" afr/afr > afr/afr2; mv afr/afr2 afr/afr

# Swedish
sed -E "/\tV;/s/;SG//" swe/swe > swe/swe2; mv swe/swe2 swe/swe
sed -E "/\tV;/s/;PL//" swe/swe > swe/swe2; mv swe/swe2 swe/swe
awk -F'\t' -v OFS='\t' '{gsub(/[;|]SG/,"",$3); gsub(/[;|]PL/,"",$3); print}' swe/swe.segmentations > swe/swe.segmentations2
mv swe/swe.segmentations2 swe/swe.segmentations

# Southern Kurdish: every line starts with a tab and ends in CRLF
perl -pe 's/^\t//; s/\r$//' sdh/sdh > sdh/sdh2; mv sdh/sdh2 sdh/sdh

# Whitespace around tags (Livvi/Veps/Karelian "PRIV ", Tajik, Chukchi, Ket, Bhojpuri)
for f in olo/olo vep/vep krl/krl tgk/tgk ckt/ckt ket/ket bra/bra; do
  perl -F'\t' -lane '$F[2] =~ s/\s*([;|])\s*/$1/g; $F[2] =~ s/^\s+|\s+$//g; print join("\t", @F)' "$f" > "$f.tmp" && mv "$f.tmp" "$f"
done

# Central Kurdish: only script-specific files exist; use the Latin-script gold + silver data (files lack final newlines; drop malformed rows)
awk -F'\t' 'NF==3 && !seen[$0]++' ckb/latn/ckb-latn.tsv ckb/latn/ckb-latn-large.tsv > ckb/ckb
# the silver data writes argument marking as ArgAB3P next to ARGAB3P
perl -pe 's/\bArg(AB|ER|NO|AC|DA)([123][SP])\b/ARG$1$2/g' ckb/ckb > ckb/ckb2; mv ckb/ckb2 ckb/ckb

# Swahili: INDF1/2/3 is the -a- present with person 1/2/3
perl -pe 's/\bINDF([123])\b/$1;PRS/g' swc/swc > swc/swc2; mv swc/swc2 swc/swc

# Old French: "not{NOM}" and "non{NOM}" are the same tag
perl -pi -e 's/\bnot\{NOM\}/non{NOM}/g' fro/fro

# Uyghur: lemmas are infinitives (achmaq), UD lemmas bare stems (ach); strip the ending so they can match
awk -F'\t' 'BEGIN{OFS="\t"} {sub(/(maq|mek)$/,"",$1); print}' uig/uig > uig/uig2; mv uig/uig2 uig/uig

touch "$MARKER"
