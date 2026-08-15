#!/bin/bash
# Start MLflow tracking server with RDS backend and S3 artifact store
set -euo pipefail

AWS_REGION="${AWS_REGION:-us-east-1}"
SECRET_NAME="mlops-pipeline/mlflow/db-credentials"

echo "================================================"
echo "Setting up MLflow Tracking Server"
echo "================================================"

SECRET=$(aws secretsmanager get-secret-value \
  --secret-id ${SECRET_NAME} \
  --region ${AWS_REGION} \
  --query SecretString \
  --output text)

DB_USERNAME=$(echo ${SECRET} | python3 -c "import sys,json; print(json.load(sys.stdin)['username'])")
DB_PASSWORD=$(echo ${SECRET} | python3 -c "import sys,json,urllib.parse; print(urllib.parse.quote(json.load(sys.stdin)['password'], safe=''))")
DB_HOST=$(echo ${SECRET} | python3 -c "import sys,json; print(json.load(sys.stdin)['host'])")
DB_NAME=$(echo ${SECRET} | python3 -c "import sys,json; print(json.load(sys.stdin)['dbname'])")

MLFLOW_BUCKET=$(aws ssm get-parameter \
  --name /mlops/mlflow-bucket \
  --region ${AWS_REGION} \
  --query Parameter.Value \
  --output text)

MLFLOW_TRACKING_URI="postgresql://${DB_USERNAME}:${DB_PASSWORD}@${DB_HOST}:5432/${DB_NAME}"

echo "Starting MLflow tracking server..."
nohup mlflow server \
  --backend-store-uri "${MLFLOW_TRACKING_URI}" \
  --default-artifact-root "s3://${MLFLOW_BUCKET}/mlflow-artifacts" \
  --host 0.0.0.0 \
  --port 5000 \
  --serve-artifacts \
  > ~/mlflow-server.log 2>&1 &

echo "Waiting for MLflow server to be ready..."
for i in $(seq 1 30); do
  if curl -sf http://localhost:5000/health > /dev/null 2>&1; then
    echo "MLflow server is ready at http://localhost:5000"
    break
  fi
  sleep 2
done

echo ""
echo "================================================"
echo "MLflow server running"
echo "Artifacts: s3://${MLFLOW_BUCKET}/mlflow-artifacts"
echo "Logs: ~/mlflow-server.log"
echo "Port-forward: ssh -L 5000:localhost:5000 -i KEY.pem ec2-user@EC2_IP"
echo "================================================"
