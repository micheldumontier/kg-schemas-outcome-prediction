#!/bin/bash
# Fetch the SPHN RDF Schema release (CC BY 4.0) from the SPHN Data Coordination Centre GitLab, pinned to a tag.
# Usage: ontologies/fetch_sphn.sh [tag]   (default 2026-1, the latest release as of October 2026)
set -euo pipefail
TAG=${1:-2026-1}
OUT=$(dirname "$0")/sphn/$TAG
B="https://git.dcc.sib.swiss/api/v4/projects/sphn-semantic-framework%2Fsphn-schema/repository"
mkdir -p "$OUT"
for d in rdf_schema quality_assurance; do
  curl -sfL "$B/archive.tar.gz?sha=$TAG&path=$d" | tar -xz -C "$OUT" --strip-components=1
done
curl -sfL -o "$OUT/LICENSE" "$B/files/LICENSE/raw?ref=$TAG"
curl -sfL -o "$OUT/README.md" "$B/files/README.md/raw?ref=$TAG"
grep -m1 "owl:versionIRI" "$OUT/rdf_schema/sphn_rdf_schema.ttl"
