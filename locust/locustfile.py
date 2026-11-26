"""Flood test: many simulated users sending leaf photos to /predict.

Headless run against the load balancer:
  uv run locust -f locust/locustfile.py --host http://localhost:8080 --headless \
      -u 50 -r 10 -t 60s --csv locust/results/c1
"""

import random
from pathlib import Path

from locust import HttpUser, between, task

TEST_DIR = Path(__file__).resolve().parent.parent / "data" / "test"
IMAGES = [(p.name, p.read_bytes()) for p in sorted(TEST_DIR.glob("*/*.jpg"))]


class FarmerUser(HttpUser):
    wait_time = between(0.5, 1.5)

    @task(10)
    def predict(self):
        name, data = random.choice(IMAGES)
        self.client.post("/predict", files={"file": (name, data, "image/jpeg")}, name="/predict")

    @task(1)
    def health(self):
        self.client.get("/health", name="/health")
