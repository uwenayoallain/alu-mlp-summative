#!/usr/bin/env bash
# Flood test with 1, 2 and 4 API containers behind nginx.
# Usage: bash scripts/run_flood_test.sh [users] [duration]
set -euo pipefail
cd "$(dirname "$0")/.."
USERS=${1:-60}
DURATION=${2:-60s}
mkdir -p locust/results

docker compose build api
for n in 1 2 4; do
  echo "=== $n container(s) ==="
  docker compose up -d --scale api="$n" --remove-orphans
  # wait until every replica answers /health through the balancer
  for _ in $(seq 1 90); do
    healthy=$(docker compose ps api --format '{{.Health}}' | grep -c healthy || true)
    [ "$healthy" -ge "$n" ] && break
    sleep 2
  done
  sleep 3
  uv run locust -f locust/locustfile.py --host http://localhost:8080 --headless \
    -u "$USERS" -r 10 -t "$DURATION" --csv "locust/results/c$n" --only-summary || true  # locust exits 1 if any request failed
done
docker compose down
uv run python scripts/summarize_flood_test.py
