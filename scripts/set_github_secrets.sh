#!/usr/bin/env bash
set -euo pipefail

# Usage: set GITHUB_REPO="owner/repo" and export env vars for secrets, then run this script.
# Example:
# GITHUB_REPO=Eaglewings966/mlops-pipeline AWS_ROLE_ARN="arn:aws:iam::123:role/Role" \ 
#   DVC_BUCKET=my-bucket MLFLOW_BUCKET=mlflow-bucket ./scripts/set_github_secrets.sh

if [ -z "${GITHUB_REPO:-}" ]; then
  echo "Error: GITHUB_REPO environment variable must be set to 'owner/repo'"
  exit 1
fi

secrets=(
  AWS_ROLE_ARN
  DVC_BUCKET
  MLFLOW_BUCKET
  EVIDENTLY_BUCKET
  SNS_TOPIC_ARN
  SERVING_HOST
  AWS_ACCESS_KEY_ID
  AWS_SECRET_ACCESS_KEY
  AWS_SESSION_TOKEN
)

for s in "${secrets[@]}"; do
  val="${!s:-}"
  if [ -z "$val" ]; then
    echo "Skipping $s: environment variable not set"
    continue
  fi
  echo "Setting secret $s in $GITHUB_REPO"
  printf '%s' "$val" | gh secret set "$s" --repo "$GITHUB_REPO"
done

echo "Done. Verify secrets with: gh secret list --repo $GITHUB_REPO"
