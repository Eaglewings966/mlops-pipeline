# setup_github_oidc.ps1
#
# EMERGENCY FALLBACK ONLY — normally Terraform manages the OIDC provider
# and IAM role trust policy (see terraform/main.tf).
#
# Only run this if you need to patch the trust policy outside of a
# Terraform apply (e.g., Terraform state is lost or role was recreated manually).
#
# Usage:
#   .\scripts\setup_github_oidc.ps1
#   .\scripts\setup_github_oidc.ps1 -AccountId 123456789012 -Repo MyOrg/my-repo

param(
  [string]$AccountId = (aws sts get-caller-identity --query Account --output text),
  [string]$Repo      = "Eaglewings966/mlops-pipeline",
  [string]$RoleName  = "mlops-pipeline-runner-role"
)

$ErrorActionPreference = "Stop"

# 1. Ensure the GitHub OIDC provider exists
$existingProviders = aws iam list-open-id-connect-providers --output json | ConvertFrom-Json
$providerArn = "arn:aws:iam::${AccountId}:oidc-provider/token.actions.githubusercontent.com"

if ($existingProviders.OpenIDConnectProviderList.Arn -notcontains $providerArn) {
  Write-Host "Creating GitHub OIDC provider..."
  aws iam create-open-id-connect-provider `
    --url https://token.actions.githubusercontent.com `
    --client-id-list sts.amazonaws.com `
    --thumbprint-list 6938fd4d98bab03faadb97b34396831e3780aea1
  Write-Host "OIDC provider created: $providerArn"
} else {
  Write-Host "OIDC provider already exists: $providerArn"
}

# 2. Update the role trust policy to include GitHub Actions + EC2
$trustPolicy = @"
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Principal": { "Service": "ec2.amazonaws.com" },
      "Action": "sts:AssumeRole"
    },
    {
      "Effect": "Allow",
      "Principal": {
        "Federated": "$providerArn"
      },
      "Action": "sts:AssumeRoleWithWebIdentity",
      "Condition": {
        "StringEquals": {
          "token.actions.githubusercontent.com:aud": "sts.amazonaws.com"
        },
        "StringLike": {
          "token.actions.githubusercontent.com:sub": "repo:${Repo}:*"
        }
      }
    }
  ]
}
"@

$tmpFile = "$env:TEMP\trust-policy.json"
$trustPolicy | Out-File -FilePath $tmpFile -Encoding utf8

Write-Host "Updating trust policy on role: $RoleName..."
aws iam update-assume-role-policy `
  --role-name $RoleName `
  --policy-document "file://$tmpFile"

Write-Host ""
Write-Host "Done. Trust policy updated for $RoleName."
Write-Host "GitHub Actions workflows in '$Repo' can now assume this role via OIDC."
