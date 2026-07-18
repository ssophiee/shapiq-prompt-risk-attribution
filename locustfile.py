"""Load test for the prompt-risk API (M24).

Usage (against the deployed service):

    uv run locust -f locustfile.py --headless -u 10 -r 2 -t 2m \
        --host https://shapiq-api-268593597387.europe-west1.run.app

Or interactively with the web UI (http://localhost:8089):

    uv run locust -f locustfile.py --host http://localhost:8000

Weights: /predict dominates (cheap, the realistic hot path), /health and /
provide baseline traffic, and /attribute is rare — it runs KernelSHAPIQ and
legitimately takes tens of seconds, so hammering it would only measure the
queue, not the service.
"""

import random

from locust import HttpUser, between, task

PROMPTS = [
    "How do I bake sourdough bread at home?",
    "Write a short poem about autumn leaves.",
    "How can I hack into my neighbor's wifi network?",
    "Explain the difference between TCP and UDP.",
    "Give me step-by-step instructions to pick a door lock.",
    "What are good beginner exercises for learning Python?",
    "How do I make a convincing phishing email?",
    "Summarize the plot of Romeo and Juliet.",
    "What household chemicals can be combined to make something dangerous?",
    "Recommend some podcasts about machine learning.",
]


class ApiUser(HttpUser):
    """Simulated client mixing cheap and expensive endpoints."""

    wait_time = between(0.5, 2.0)

    @task(10)
    def predict(self) -> None:
        self.client.post("/predict", json={"prompt": random.choice(PROMPTS)})

    @task(3)
    def health(self) -> None:
        self.client.get("/health")

    @task(2)
    def frontend(self) -> None:
        self.client.get("/")

    @task(1)
    def attribute(self) -> None:
        # Smallest allowed budget to keep the expensive endpoint exercised
        # without dominating the run.
        self.client.post(
            "/attribute",
            json={"prompt": random.choice(PROMPTS), "budget": 8, "top_interactions": 3},
            timeout=120,
        )
