param(
  [string]$GitHubRepo
)

if (-not $GitHubRepo) {
  Write-Error "Parameter -GitHubRepo is required (format: owner/repo)"
  exit 1
}

$secrets = @(
  'AWS_ROLE_ARN', 'DVC_BUCKET', 'MLFLOW_BUCKET', 'EVIDENTLY_BUCKET', 'SNS_TOPIC_ARN', 'SERVING_HOST',
  'AWS_ACCESS_KEY_ID', 'AWS_SECRET_ACCESS_KEY', 'AWS_SESSION_TOKEN'
)

foreach ($s in $secrets) {
  $val = $env:$s
  if (-not $val) {
    Write-Host "Skipping $s: environment variable not set"
    continue
  }
  Write-Host "Setting secret $s in $GitHubRepo"
  $val | gh secret set $s --repo $GitHubRepo
}

Write-Host "Done. Verify with: gh secret list --repo $GitHubRepo"
