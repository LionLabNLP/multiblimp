#!/usr/bin/env bash
# Usage: scripts/benchmark/eval_goldfish.sh [gpu] [--dry-run] [-- extra eval_model.py args]
# Scores each Goldfish model on its own language only, one after the other.
# Languages without output/benchmark/<Language>.parquet are skipped.
set -euo pipefail
cd "$(dirname "$0")/../.."

GPU=${1:-0}; shift $(( $# > 0 ? 1 : 0 ))
DRY=""; [[ ${1:-} == --dry-run ]] && { DRY=1; shift; }
[[ ${1:-} == -- ]] && shift
EXTRA=("$@")

# <model> <language as in output/benchmark/<Language>.parquet>
MODELS="
goldfish-models/abk_cyrl_10mb Abkhaz
goldfish-models/afr_latn_1000mb Afrikaans
goldfish-models/sqi_latn_1000mb Albanian
goldfish-models/gsw_latn_100mb Alemannic
goldfish-models/amh_ethi_1000mb Amharic
goldfish-models/grc_grek_100mb Ancient_Greek
goldfish-models/arb_arab_1000mb Arabic
goldfish-models/hye_armn_1000mb Armenian
goldfish-models/asm_beng_100mb Assamese
goldfish-models/aze_latn_1000mb Azerbaijani
goldfish-models/bam_latn_10mb Bambara
goldfish-models/eus_latn_1000mb Basque
goldfish-models/bar_latn_10mb Bavarian
goldfish-models/bel_cyrl_1000mb Belarusian
goldfish-models/ben_beng_1000mb Bengali
goldfish-models/bho_deva_10mb Bhojpuri
goldfish-models/bre_latn_100mb Breton
goldfish-models/bul_cyrl_1000mb Bulgarian
goldfish-models/bxr_cyrl_10mb Buryat
goldfish-models/yue_hant_10mb Cantonese
goldfish-models/cat_latn_1000mb Catalan
goldfish-models/ceb_latn_100mb Cebuano
goldfish-models/ckb_arab_100mb Central_Kurdish
goldfish-models/zho_hans_1000mb Chinese
goldfish-models/lzh_hant_10mb Classical_Chinese
goldfish-models/hrv_latn_1000mb Croatian
goldfish-models/ces_latn_1000mb Czech
goldfish-models/dan_latn_1000mb Danish
goldfish-models/nld_latn_1000mb Dutch
goldfish-models/eng_latn_1000mb English
goldfish-models/myv_cyrl_10mb Erzya
goldfish-models/epo_latn_1000mb Esperanto
goldfish-models/est_latn_1000mb Estonian
goldfish-models/fao_latn_100mb Faroese
goldfish-models/fin_latn_1000mb Finnish
goldfish-models/fra_latn_1000mb French
goldfish-models/glg_latn_1000mb Galician
goldfish-models/kat_geor_1000mb Georgian
goldfish-models/deu_latn_1000mb German
goldfish-models/ell_grek_1000mb Greek
goldfish-models/guj_gujr_1000mb Gujarati
goldfish-models/hat_latn_100mb Haitian_Creole
goldfish-models/hau_latn_1000mb Hausa
goldfish-models/heb_hebr_1000mb Hebrew
goldfish-models/hin_deva_1000mb Hindi
goldfish-models/hun_latn_1000mb Hungarian
goldfish-models/isl_latn_1000mb Icelandic
goldfish-models/ind_latn_1000mb Indonesian
goldfish-models/gle_latn_100mb Irish
goldfish-models/ita_latn_1000mb Italian
goldfish-models/jpn_jpan_1000mb Japanese
goldfish-models/jav_latn_100mb Javanese
goldfish-models/kas_deva_5mb Kashmiri
goldfish-models/kaz_cyrl_1000mb Kazakh
goldfish-models/kpv_cyrl_5mb Komi_Zyrian
goldfish-models/kor_hang_1000mb Korean
goldfish-models/kir_cyrl_1000mb Kyrgyz
goldfish-models/ltg_latn_10mb Latgalian
goldfish-models/lat_latn_1000mb Latin
goldfish-models/lav_latn_1000mb Latvian
goldfish-models/lij_latn_10mb Ligurian
goldfish-models/lit_latn_1000mb Lithuanian
goldfish-models/nds_latn_10mb Low_Saxon
goldfish-models/ltz_latn_100mb Luxembourgish
goldfish-models/mkd_cyrl_1000mb Macedonian
goldfish-models/mal_mlym_1000mb Malayalam
goldfish-models/mlt_latn_1000mb Maltese
goldfish-models/glv_latn_5mb Manx
goldfish-models/mar_deva_1000mb Marathi
goldfish-models/mdf_cyrl_5mb Moksha
goldfish-models/nap_latn_5mb Neapolitan
goldfish-models/nep_deva_1000mb Nepali
goldfish-models/sme_latn_10mb North_Sami
goldfish-models/kmr_latn_10mb Northern_Kurdish
goldfish-models/nor_latn_1000mb Norwegian
goldfish-models/oci_latn_100mb Occitan
goldfish-models/ori_orya_100mb Odia
goldfish-models/ang_latn_5mb Old_English
goldfish-models/pus_arab_1000mb Pashto
goldfish-models/fas_arab_1000mb Persian
goldfish-models/pol_latn_1000mb Polish
goldfish-models/por_latn_1000mb Portuguese
goldfish-models/pan_guru_1000mb Punjabi
goldfish-models/ron_latn_1000mb Romanian
goldfish-models/rus_cyrl_1000mb Russian
goldfish-models/san_deva_100mb Sanskrit
goldfish-models/srd_latn_10mb Sardinian
goldfish-models/gla_latn_100mb Scottish_Gaelic
goldfish-models/srp_latn_1000mb Serbian
goldfish-models/wuu_hani_10mb Shanghainese
goldfish-models/scn_latn_100mb Sicilian
goldfish-models/snd_arab_100mb Sindhi
goldfish-models/sin_sinh_1000mb Sinhala
goldfish-models/slk_latn_1000mb Slovak
goldfish-models/slv_latn_1000mb Slovenian
goldfish-models/spa_latn_1000mb Spanish
goldfish-models/swe_latn_1000mb Swedish
goldfish-models/tgl_latn_1000mb Tagalog
goldfish-models/tam_taml_1000mb Tamil
goldfish-models/tat_cyrl_1000mb Tatar
goldfish-models/tel_telu_1000mb Telugu
goldfish-models/tha_thai_1000mb Thai
goldfish-models/tir_ethi_100mb Tigrinya
goldfish-models/tsn_latn_100mb Tswana
goldfish-models/tur_latn_1000mb Turkish
goldfish-models/ukr_cyrl_1000mb Ukrainian
goldfish-models/hsb_latn_10mb Upper_Sorbian
goldfish-models/urd_arab_1000mb Urdu
goldfish-models/uig_arab_100mb Uyghur
goldfish-models/uzb_latn_1000mb Uzbek
goldfish-models/vep_latn_5mb Veps
goldfish-models/vie_latn_1000mb Vietnamese
goldfish-models/cym_latn_1000mb Welsh
goldfish-models/wol_latn_10mb Wolof
goldfish-models/sah_cyrl_100mb Yakut
goldfish-models/yid_hebr_100mb Yiddish
goldfish-models/yor_latn_100mb Yoruba
goldfish-models/zza_latn_10mb Zazaki
# FALLBACKS: no Goldfish model exists for these languages; each uses the model of the closest language/variety
goldfish-models/hye_armn_1000mb Western_Armenian
goldfish-models/hye_armn_1000mb Classical_Armenian
goldfish-models/hye_armn_1000mb Middle_Armenian
goldfish-models/kpv_cyrl_5mb Komi_Permyak
goldfish-models/sme_latn_10mb Skolt_Sami
goldfish-models/apc_arab_5mb South_Levantine_Arabic
goldfish-models/ckb_arab_100mb Southern_Kurdish
goldfish-models/grn_latn_10mb Mbya_Guarani
goldfish-models/syr_syrc_10mb Assyrian
goldfish-models/fra_latn_1000mb Old_French
goldfish-models/fra_latn_1000mb Middle_French
goldfish-models/bul_cyrl_1000mb Old_Church_Slavonic
goldfish-models/rus_cyrl_1000mb Old_East_Slavic
goldfish-models/tur_latn_1000mb Ottoman_Turkish
goldfish-models/heb_hebr_1000mb Ancient_Hebrew
goldfish-models/ell_grek_1000mb Cappadocian
goldfish-models/bul_cyrl_1000mb Pomak
"

SKIPPED=()
while read -r model lang; do
  [[ -z $model || $model == \#* ]] && continue
  if [[ ! -f output/benchmark/$lang.parquet ]]; then
    SKIPPED+=("$lang")
    continue
  fi
  if [[ -n $DRY ]]; then
    echo "./eval.sh $model $GPU --langs $lang ${EXTRA[*]-}"
  else
    WAIT=1 ./eval.sh "$model" "$GPU" --langs "$lang" ${EXTRA[@]+"${EXTRA[@]}"} || echo "FAILED $model ($lang)" >&2
  fi
done <<< "$MODELS"

(( ${#SKIPPED[@]} )) && echo "not in benchmark, skipped: ${SKIPPED[*]}" >&2
exit 0
