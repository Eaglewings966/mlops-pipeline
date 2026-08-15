#!/bin/bash
# Destroy all MLOps pipeline infrastructure
set -euo pipefail

AWS_REGION="us-east-1"

echo "================================================"
echo "Destroying MLOps Pipeline Infrastructure"
echo "================================================"

read -p "Type 'destroy' to confirm: " confirm
if [ "${confirm}" != "destroy" ]; then
  echo "Cancelled"
  exit 0
fi

# Stop running processes
echo "Stopping API and MLflow servers..."
pkill -f uvicorn 2>/dev/null || true
pkill -f mlflow 2>/dev/null || true

# Empty S3 buckets
for bucket_ssm in /mlops/dvc-bucket /mlops/mlflow-bucket /mlops/evidently-bucket; do
  BUCKET=$(aws ssm get-parameter \
    --name ${bucket_ssm} \
    --region ${AWS_REGION} \
    --query Parameter.Value \
    --output text 2>/dev/null || echo "")

  if [ -n "${BUCKET}" ]; then
    echo "Emptying bucket: ${BUCKET}"
    aws s3 rm s3://${BUCKET} --recursive --region ${AWS_REGION} 2>/dev/null || true
  fi
done

# Terraform destroy
echo "Running terraform destroy..."
cd terraform
terraform destroy --auto-approve
cd ..

# Terminate EC2 instances tagged with this project
INSTANCE_IDS=$(aws ec2 describe-instances \
  --filters "Name=tag:Project,Values=mlops-pipeline" \
            "Name=instance-state-name,Values=running" \
  --query 'Reservations[*].Instances[*].InstanceId' \
  --output text \
  --region ${AWS_REGION})

if [ -n "${INSTANCE_IDS}" ]; then
  aws ec2 terminate-instances --instance-ids ${INSTANCE_IDS} --region ${AWS_REGION}
  echo "Terminated EC2 instances: ${INSTANCE_IDS}"
fi

echo ""
echo "================================================"
echo "Destroy complete. Verify in AWS console."
echo "================================================"
