$trustPolicy = @'
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
        "Federated": "arn:aws:iam::047423858035:oidc-provider/token.actions.githubusercontent.com"
      },
      "Action": "sts:AssumeRoleWithWebIdentity",
      "Condition": {
        "StringLike": {
          "token.actions.githubusercontent.com:sub": "repo:Eaglewings966/mlops-pipeline:*"
        },
        "StringEquals": {
          "token.actions.githubusercontent.com:aud": "sts.amazonaws.com"
        }
      }
    }
  ]
}
'@

$trustPolicy | Out-File -FilePath "$env:TEMP\trust-policy.json" -Encoding utf8

aws iam update-assume-role-policy `
  --role-name mlops-pipeline-runner-role `
  --policy-document file://$env:TEMP\trust-policy.json

Write-Host "Trust policy updated successfully"
