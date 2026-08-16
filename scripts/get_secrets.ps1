param(
  [string]$Region = "us-east-1",
  [string]$OutputFile = ".env",
  [string]$GitHubRepo = "Eaglewings966/mlops-pipeline",
  [switch]$SetGitHubSecrets
)

$ErrorActionPreference = "Stop"

Write-Host "Fetching pipeline secrets from AWS ($Region)..."

$ssmNames = @(
  "/mlops/dvc-bucket",
  "/mlops/mlflow-bucket",
  "/mlops/evidently-bucket",
  "/mlops/drift-sns-arn"
)

$ssmJson = aws ssm get-parameters `
  --names $ssmNames `
  --region $Region `
  --output json | ConvertFrom-Json

$ssm = @{}
foreach ($param in $ssmJson.Parameters) {
  $ssm[$param.Name] = $param.Value
}

$hostIp = aws ec2 describe-instances `
  --filters "Name=tag:Project,Values=mlops-pipeline" "Name=instance-state-name,Values=running" `
  --query "Reservations[0].Instances[0].PublicIpAddress" `
  --output text `
  --region $Region

if (-not $hostIp -or $hostIp -eq "None") {
  Write-Error "No running mlops-pipeline EC2 instance found."
}

$roleArn = aws iam get-role `
  --role-name mlops-pipeline-runner-role `
  --query "Role.Arn" `
  --output text

$values = [ordered]@{
  AWS_REGION              = $Region
  DVC_BUCKET              = $ssm["/mlops/dvc-bucket"]
  MLFLOW_BUCKET           = $ssm["/mlops/mlflow-bucket"]
  EVIDENTLY_BUCKET        = $ssm["/mlops/evidently-bucket"]
  SNS_TOPIC_ARN           = $ssm["/mlops/drift-sns-arn"]
  MLFLOW_HOST             = $hostIp
  SERVING_HOST            = $hostIp
  AWS_ROLE_ARN            = $roleArn
  MLFLOW_TRACKING_URI     = "http://${hostIp}:5000"
  MLFLOW_EXPERIMENT_NAME  = "fraud-detection"
  MLFLOW_MODEL_NAME       = "fraud-detection-xgboost"
  MLFLOW_DB_SECRET_NAME   = "mlops-pipeline/mlflow/db-credentials"
  MODEL_STAGE             = "Production"
  SERVING_PORT            = "8000"
  DATA_DIR                = "data"
}

Write-Host ""
Write-Host "Pipeline secrets:"
foreach ($entry in $values.GetEnumerator()) {
  Write-Host ("  {0}={1}" -f $entry.Key, $entry.Value)
}

$envLines = @()
foreach ($entry in $values.GetEnumerator()) {
  $envLines += "{0}={1}" -f $entry.Key, $entry.Value
}
$envLines | Set-Content -Path $OutputFile -Encoding utf8
Write-Host ""
Write-Host "Wrote $OutputFile"

foreach ($entry in $values.GetEnumerator()) {
  Set-Item -Path "env:$($entry.Key)" -Value $entry.Value
}

if ($SetGitHubSecrets) {
  Write-Host ""
  Write-Host "Setting GitHub secrets in $GitHubRepo..."
  $githubSecrets = @(
    "AWS_ROLE_ARN", "DVC_BUCKET", "MLFLOW_BUCKET", "EVIDENTLY_BUCKET",
    "SNS_TOPIC_ARN", "MLFLOW_HOST", "SERVING_HOST"
  )
  foreach ($name in $githubSecrets) {
    $val = $values[$name]
    if (-not $val) {
      Write-Host "Skipping ${name}: no value"
      continue
    }
    Write-Host "Setting $name"
    $val | gh secret set $name --repo $GitHubRepo
  }
  Write-Host "Done. Verify with: gh secret list --repo $GitHubRepo"
}
