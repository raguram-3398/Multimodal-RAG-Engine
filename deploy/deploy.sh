#!/usr/bin/env bash
# Run on the EC2 instance: ./deploy.sh <full commit SHA>
# Pulls that image from ECR and replaces the running container.
set -euo pipefail

REGION="us-east-2"
REGISTRY="171022098864.dkr.ecr.us-east-2.amazonaws.com"
IMAGE="$REGISTRY/multimodal-rag-engine"
PARAM="/multimodal-rag-engine/pinecone-api-key"
CONTAINER="multimodal-rag-engine"
LOG_GROUP="/multimodal-rag-engine/app"
PORT=7860
AWS_TIMEOUTS=(--cli-connect-timeout 10 --cli-read-timeout 30)
PULL_TIMEOUT_S=600

SHA="${1:?usage: deploy.sh <full commit SHA>}"

aws ecr get-login-password --region "$REGION" "${AWS_TIMEOUTS[@]}" |
  docker login --username AWS --password-stdin "$REGISTRY"
timeout "$PULL_TIMEOUT_S" docker pull "$IMAGE:$SHA"

PINECONE_API_KEY="$(aws ssm get-parameter --region "$REGION" "${AWS_TIMEOUTS[@]}" \
  --name "$PARAM" --with-decryption --query Parameter.Value --output text)"
export PINECONE_API_KEY

docker rm -f "$CONTAINER" 2>/dev/null || true
docker run -d --name "$CONTAINER" --restart unless-stopped \
  --log-driver awslogs --log-opt awslogs-region="$REGION" \
  --log-opt awslogs-group="$LOG_GROUP" --log-opt awslogs-stream="$SHA" \
  -p "$PORT:$PORT" -e PINECONE_API_KEY "$IMAGE:$SHA"
echo "running $IMAGE:$SHA"
