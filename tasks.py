import os

from invoke import Context, task

WINDOWS = os.name == "nt"
PROJECT_NAME = "shapiq_attribution"
PYTHON_VERSION = "3.13"


# Project commands
@task
def preprocess_data(ctx: Context) -> None:
    """Build the processed prompt-risk dataset from raw snapshots."""
    ctx.run(
        (
            "uv run python -m shapiq_attribution.data build-dataset "
            "--raw-dir data/raw "
            "--output-path data/processed/prompt_risk_dataset.jsonl"
        ),
        echo=True,
        pty=not WINDOWS,
    )


@task
def train(ctx: Context) -> None:
    """Train model."""
    ctx.run("uv run python -m shapiq_attribution.train", echo=True, pty=not WINDOWS)


@task
def test(ctx: Context) -> None:
    """Run tests."""
    ctx.run("uv run coverage run -m pytest tests/", echo=True, pty=not WINDOWS)
    ctx.run("uv run coverage report -m -i", echo=True, pty=not WINDOWS)

@task
def build_baseline(ctx: Context) -> None:
    """Snapshot the training-set feature distribution for drift monitoring."""
    ctx.run(
        "uv run python -m shapiq_attribution.monitoring baseline",
        echo=True,
        pty=not WINDOWS,
    )


@task
def monitor_report(ctx: Context) -> None:
    """Generate the Evidently drift + health report from logged predictions."""
    ctx.run(
        "uv run python -m shapiq_attribution.monitoring report",
        echo=True,
        pty=not WINDOWS,
    )


@task
def docker_build_train(ctx: Context, progress: str = "plain") -> None:
    """Build the training Docker image."""
    ctx.run(
        f"docker build -t shapiq-train:latest . -f dockerfiles/train.dockerfile --progress={progress}",
        echo=True,
        pty=not WINDOWS,
    )


@task
def docker_build_train_gpu(ctx: Context, progress: str = "plain") -> None:
    """Build the GPU training Docker image."""
    ctx.run(
        f"docker build -t shapiq-train-gpu:latest . -f dockerfiles/train.gpu.dockerfile --progress={progress}",
        echo=True,
        pty=not WINDOWS,
    )


@task
def docker_build_api(ctx: Context, progress: str = "plain") -> None:
    """Build the API (serving) Docker image."""
    ctx.run(
        f"docker build -t shapiq-api:latest . -f dockerfiles/api.dockerfile --progress={progress}",
        echo=True,
        pty=not WINDOWS,
    )


@task
def docker_run_api(ctx: Context, port: int = 8000) -> None:
    """Run the API container, mounting the local model directory read-only."""
    ctx.run(
        f'docker run --rm -p {port}:8000 -v "$PWD/models:/app/models:ro" shapiq-api:latest',
        echo=True,
        pty=not WINDOWS,
    )


@task
def docker_build(ctx: Context, progress: str = "plain") -> None:
    """Build all Docker images (train + api)."""
    docker_build_train(ctx, progress=progress)
    docker_build_api(ctx, progress=progress)


@task
def compose_up(ctx: Context, service: str = "api") -> None:
    """Start a service via docker compose (default: api)."""
    ctx.run(f"docker compose up --build {service}", echo=True, pty=not WINDOWS)


@task
def compose_down(ctx: Context) -> None:
    """Stop and remove docker compose services."""
    ctx.run("docker compose down", echo=True, pty=not WINDOWS)


# Documentation commands
@task
def build_docs(ctx: Context) -> None:
    """Build documentation."""
    ctx.run("uv run mkdocs build --config-file docs/mkdocs.yaml --site-dir build", echo=True, pty=not WINDOWS)


@task
def serve_docs(ctx: Context) -> None:
    """Serve documentation."""
    ctx.run("uv run mkdocs serve --config-file docs/mkdocs.yaml", echo=True, pty=not WINDOWS)
