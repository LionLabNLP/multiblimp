#!/usr/bin/env bash
set -euxo pipefail

cd "$(dirname "$0")"
UD_VERSION="$(tr -d '[:space:]' < UD_VERSION)"

# Release handle: http://hdl.handle.net/11234/1-6316 (UD 2.19; 2.18 was 11234/1-6149). Update both on a new release.
UD_HANDLE="11234/1-6316"
UD_URL="https://lindat.mff.cuni.cz/repository/server/api/core/bitstreams/handle/${UD_HANDLE}/ud-treebanks-v${UD_VERSION}.tgz"

curl -o "ud-treebanks-v${UD_VERSION}.tgz" "$UD_URL"

tar -xvzf "ud-treebanks-v${UD_VERSION}.tgz"

# Get the not-to-release data for Yupik-SLI, to avoid issues with morpheme segmentation.
curl -L -o "./ud-treebanks-v${UD_VERSION}/UD_Yupik-SLI/ess_sli-ud-test.conllu" \
  "https://raw.githubusercontent.com/UniversalDependencies/UD_Yupik-SLI/master/not-to-release/ess_sli-ud-test.merged.conllu"