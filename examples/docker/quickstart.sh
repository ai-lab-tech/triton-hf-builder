#!/bin/sh
set -eu

cd "$(dirname "$0")"
export DEMO_UID="$(id -u)"
export DEMO_GID="$(id -g)"
mkdir -p data/hf-cache

docker compose build prepare
docker compose run --rm --no-deps prepare
cp data/model_repository/requirements.txt requirements.generated.txt
docker compose build triton
if ! docker compose up --detach --wait --wait-timeout 900 triton; then
    docker compose logs --tail 100 triton
    exit 1
fi
docker compose exec -T triton python3 /opt/demo/infer.py
