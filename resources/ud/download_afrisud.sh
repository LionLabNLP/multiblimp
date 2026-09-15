cd ./ud-treebanks-v2.18
urls=(
  "https://gmd-universal.grew.fr/tgz/UD_Hausa-AfriSUD@conv.tgz"
  "https://gmd-universal.grew.fr/tgz/UD_isiXhosa-AfriSUD@conv.tgz"
  "https://gmd-universal.grew.fr/tgz/UD_Kinyarwanda-AfriSUD@conv.tgz"
  "https://gmd-universal.grew.fr/tgz/UD_Naija-AfriSUD@conv.tgz"
  "https://gmd-universal.grew.fr/tgz/UD_Runyankore-AfriSUD@conv.tgz"
  "https://gmd-universal.grew.fr/tgz/UD_Swahili-AfriSUD@conv.tgz"
  "https://gmd-universal.grew.fr/tgz/UD_Wolof-AfriSUD@conv.tgz"
  "https://gmd-universal.grew.fr/tgz/UD_Yoruba-AfriSUD@conv.tgz"
)
for url in "${urls[@]}"; do
  curl -O "$url"
done
for f in *.tgz *.tar; do
  dir="${f%.tgz}"
  dir="${dir%.tar}"
  dir="${dir%@*}"
  mkdir -p "$dir"
  tmp="$(mktemp -d)"
  tar -xvzf "$f" -C "$tmp" 2>/dev/null || tar -xvf "$f" -C "$tmp"
  if [ "$(find "$tmp" -mindepth 1 -maxdepth 1 -type d | wc -l)" -eq 1 ] && [ "$(find "$tmp" -mindepth 1 -maxdepth 1 -type f | wc -l)" -eq 0 ]; then
    inner="$(find "$tmp" -mindepth 1 -maxdepth 1 -type d)"
    mv "$inner"/* "$dir"/
  else
    mv "$tmp"/* "$dir"/
  fi
  rm -rf "$tmp"
  rm "$f"
done