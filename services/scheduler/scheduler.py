#!/usr/bin/env python3
"""Data update scheduler with first-run bootstrap and recurring pipeline runs."""

import os
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import docker
from docker.errors import DockerException
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.interval import IntervalTrigger

SCHEDULER_INTERVAL_DAYS = int(os.getenv("SCHEDULER_INTERVAL_DAYS", "14"))
RUN_ON_STARTUP = os.getenv("RUN_ON_STARTUP", "true").lower() == "true"
STATE_DIR = os.getenv("SCHEDULER_STATE_DIR", "/state")
BOOTSTRAP_MARKER = os.getenv("BOOTSTRAP_MARKER", f"{STATE_DIR}/bootstrap_complete")
QDRANT_HEALTH_URL = os.getenv("QDRANT_HEALTH_URL", "http://qdrant-db:6333/collections")
QDRANT_STARTUP_TIMEOUT_SECONDS = int(os.getenv("QDRANT_STARTUP_TIMEOUT_SECONDS", "300"))
PIPELINE_NETWORK = os.getenv("PIPELINE_NETWORK", "suchemacschine_net")
DATA_VOLUME = os.getenv("PIPELINE_DATA_VOLUME", "suchemacschine_scrape_data")
PDF_VOLUME = os.getenv("PIPELINE_PDF_VOLUME", "suchemacschine_pdf_sources")
CONTAINER_RUNTIME_SOCKET = os.getenv("CONTAINER_RUNTIME_SOCKET", "unix:///var/run/docker.sock")
PARALLEL_SCRAPER_STEPS = {
    "website-scraper",
    "timetable-scraper",
}

SERVICE_PIPELINE = [
    {
        "name": "website-scraper",
        "image": os.getenv("WEBSITE_SCRAPER_IMAGE", "suchemacschine/website-scraper:latest"),
        "description": "Scraping websites",
        "volumes": {
            DATA_VOLUME: {"bind": "/data", "mode": "rw"},
        },
    },
    {
        "name": "pdf-indexer",
        "image": os.getenv("PDF_INDEXER_IMAGE", "suchemacschine/pdf-indexer:latest"),
        "description": "Indexing PDFs",
        "volumes": {
            DATA_VOLUME: {"bind": "/data", "mode": "rw"},
            PDF_VOLUME: {"bind": "/pdf_sources", "mode": "rw"},
        },
    },
    {
        "name": "timetable-scraper",
        "image": os.getenv("TIMETABLE_SCRAPER_IMAGE", "suchemacschine/timetable-scraper:latest"),
        "description": "Scraping timetables",
        "volumes": {
            DATA_VOLUME: {"bind": "/data", "mode": "rw"},
        },
    },
    {
        "name": "text-processor",
        "image": os.getenv("TEXT_PROCESSOR_IMAGE", "suchemacschine/text-processor:latest"),
        "description": "Processing text",
        "volumes": {
            DATA_VOLUME: {"bind": "/data", "mode": "rw"},
        },
    },
    {
        "name": "embeddings-service",
        "image": os.getenv("EMBEDDINGS_SERVICE_IMAGE", "suchemacschine/embeddings-service:latest"),
        "description": "Generating embeddings",
        "volumes": {
            DATA_VOLUME: {"bind": "/data", "mode": "rw"},
        },
    },
    {
        "name": "qdrant-indexer",
        "image": os.getenv("QDRANT_INDEXER_IMAGE", "suchemacschine/qdrant-indexer:latest"),
        "description": "Indexing in Qdrant",
        "volumes": {
            DATA_VOLUME: {"bind": "/data", "mode": "rw"},
        },
    },
]

client = None


def get_container_client() -> Optional[docker.DockerClient]:
    """Create/reuse a container runtime client with Docker/Podman socket fallbacks."""
    global client

    if client is not None:
        try:
            client.ping()
            return client
        except Exception:
            client = None

    configured_host = os.getenv("DOCKER_HOST", "").strip()
    candidate_hosts = []
    if configured_host:
        candidate_hosts.append(configured_host)

    candidate_hosts.extend([
        CONTAINER_RUNTIME_SOCKET,
        "unix:///var/run/docker.sock",
        "unix:///run/user/0/podman/podman.sock",
        "unix:///run/podman/podman.sock",
    ])

    # Keep order stable while removing duplicates.
    deduped_hosts = []
    for host in candidate_hosts:
        if host and host not in deduped_hosts:
            deduped_hosts.append(host)

    last_error = None
    for host in deduped_hosts:
        try:
            probe_client = docker.DockerClient(base_url=host)
            probe_client.ping()
            client = probe_client
            print(f"Container runtime connected via {host}")
            return client
        except Exception as exc:
            last_error = exc

    print(f"Container runtime unavailable. Tried: {deduped_hosts}")
    if last_error is not None:
        print(f"Last container runtime error: {last_error}")
    return None


def resolve_local_image_name(image_name: str) -> str:
    """Prefer already-available local images, including Podman localhost-prefixed tags."""
    runtime_client = get_container_client()
    if runtime_client is None:
        return image_name

    candidates = [image_name]
    if not image_name.startswith("localhost/"):
        candidates.append(f"localhost/{image_name}")

    for candidate in candidates:
        try:
            runtime_client.images.get(candidate)
            return candidate
        except Exception:
            continue

    return image_name


def wait_for_qdrant_health() -> bool:
    """Wait until Qdrant health endpoint responds successfully."""
    import requests

    print(f"Waiting for Qdrant: {QDRANT_HEALTH_URL}")
    deadline = time.time() + QDRANT_STARTUP_TIMEOUT_SECONDS

    while time.time() < deadline:
        try:
            response = requests.get(QDRANT_HEALTH_URL, timeout=5)
            if response.status_code == 200:
                print("Qdrant is healthy")
                return True
        except Exception:
            pass

        print("Qdrant not ready yet, retrying in 5 seconds...")
        time.sleep(5)

    print("Qdrant did not become healthy before timeout")
    return False


def run_step(step: dict) -> bool:
    """Run one service step in the pipeline as an ephemeral container."""
    runtime_client = get_container_client()
    if runtime_client is None:
        print("Step aborted: container runtime is unavailable")
        return False

    service_name = step["name"]
    image_name = step["image"]
    resolved_image_name = resolve_local_image_name(image_name)
    image_candidates = [resolved_image_name]
    if not resolved_image_name.startswith("localhost/"):
        image_candidates = [f"localhost/{resolved_image_name}", resolved_image_name]

    # Keep order stable while removing duplicates.
    deduped_candidates = []
    for candidate in image_candidates:
        if candidate not in deduped_candidates:
            deduped_candidates.append(candidate)
    volumes = step.get("volumes", {})

    print(f"\n-> {step['description']} ({service_name})")
    print(f"Using image candidates: {deduped_candidates}")

    last_error = None

    for candidate in deduped_candidates:
        container = None
        try:
            print(f"Trying image: {candidate}")
            container = runtime_client.containers.run(
                image=candidate,
                name=f"{service_name}_run_{int(datetime.now().timestamp())}",
                volumes=volumes,
                network=PIPELINE_NETWORK,
                remove=False,
                detach=True,
            )

            result = container.wait()
            exit_code = result.get("StatusCode", 1)

            logs = container.logs(stdout=True, stderr=True).decode("utf-8", errors="ignore").strip()
            if logs:
                print(logs)

            if exit_code != 0:
                print(f"Step failed with exit code {exit_code}: {service_name}")
                return False

            print(f"Step completed: {service_name}")
            return True
        except Exception as exc:
            last_error = exc
            print(f"Step error with image {candidate}: {exc}")
        finally:
            if container is not None:
                try:
                    container.remove(force=True)
                except Exception:
                    pass

    print(f"Step error in {service_name}: {last_error}")
    return False


def start_step_container(step: dict):
    """Start one step container asynchronously and return the container object."""
    runtime_client = get_container_client()
    if runtime_client is None:
        print("Parallel step start aborted: container runtime is unavailable")
        return None

    service_name = step["name"]
    image_name = step["image"]
    resolved_image_name = resolve_local_image_name(image_name)
    image_candidates = [resolved_image_name]
    if not resolved_image_name.startswith("localhost/"):
        image_candidates = [f"localhost/{resolved_image_name}", resolved_image_name]

    deduped_candidates = []
    for candidate in image_candidates:
        if candidate not in deduped_candidates:
            deduped_candidates.append(candidate)

    volumes = step.get("volumes", {})
    last_error = None

    print(f"\n-> {step['description']} ({service_name}) [parallel start]")
    print(f"Using image candidates: {deduped_candidates}")

    for candidate in deduped_candidates:
        try:
            print(f"Trying image: {candidate}")
            container = runtime_client.containers.run(
                image=candidate,
                name=f"{service_name}_run_{int(datetime.now().timestamp())}",
                volumes=volumes,
                network=PIPELINE_NETWORK,
                remove=False,
                detach=True,
            )
            return container
        except Exception as exc:
            last_error = exc
            print(f"Step start error with image {candidate}: {exc}")

    print(f"Step failed to start ({service_name}): {last_error}")
    return None


def wait_for_step_container(step: dict, container) -> bool:
    """Wait for a started container, print logs, and clean up."""
    service_name = step["name"]
    try:
        result = container.wait()
        exit_code = result.get("StatusCode", 1)

        logs = container.logs(stdout=True, stderr=True).decode("utf-8", errors="ignore").strip()
        if logs:
            print(logs)

        if exit_code != 0:
            print(f"Step failed with exit code {exit_code}: {service_name}")
            return False

        print(f"Step completed: {service_name}")
        return True
    except Exception as exc:
        print(f"Step error while waiting ({service_name}): {exc}")
        return False
    finally:
        try:
            container.remove(force=True)
        except Exception:
            pass


def run_parallel_steps(steps: list) -> bool:
    """Run multiple step containers in parallel and wait for all to finish."""
    running = []

    for step in steps:
        container = start_step_container(step)
        if container is None:
            for _, running_container in running:
                try:
                    running_container.remove(force=True)
                except Exception:
                    pass
            return False
        running.append((step, container))

    all_ok = True
    for step, container in running:
        if not wait_for_step_container(step, container):
            all_ok = False

    return all_ok


def run_pipeline() -> bool:
    """Execute the full data pipeline and return True on success."""
    print(f"\n{'=' * 60}")
    print(f"Starting data pipeline at {datetime.now().isoformat()}")
    print(f"{'=' * 60}")

    if not wait_for_qdrant_health():
        print("Pipeline aborted: Qdrant is unavailable")
        return False

    if get_container_client() is None:
        print("Pipeline aborted: container runtime is unavailable")
        return False

    scraper_steps = [step for step in SERVICE_PIPELINE if step["name"] in PARALLEL_SCRAPER_STEPS]
    downstream_steps = [step for step in SERVICE_PIPELINE if step["name"] not in PARALLEL_SCRAPER_STEPS]

    if scraper_steps:
        print("\nRunning scraper stage in parallel...")
        if not run_parallel_steps(scraper_steps):
            print("Pipeline aborted due to failing parallel scraper stage")
            return False

    for step in downstream_steps:
        if not run_step(step):
            print("Pipeline aborted due to failing step")
            return False

    print(f"\n{'=' * 60}")
    print(f"Pipeline completed successfully at {datetime.now().isoformat()}")
    print(f"{'=' * 60}\n")
    return True


def run_startup_bootstrap_if_needed() -> None:
    """Run one initial bootstrap after a fresh setup."""
    marker = Path(BOOTSTRAP_MARKER)
    marker.parent.mkdir(parents=True, exist_ok=True)

    if not RUN_ON_STARTUP:
        print("Startup bootstrap disabled by RUN_ON_STARTUP=false")
        return

    if marker.exists():
        print(f"Bootstrap marker exists ({marker}), skipping first-run pipeline")
        return

    print("First startup detected, executing bootstrap pipeline")
    if run_pipeline():
        marker.write_text(datetime.now().isoformat(), encoding="utf-8")
        print(f"Bootstrap marker written: {marker}")
    else:
        print("Bootstrap pipeline failed, marker not written")


def main() -> None:
    """Initialize scheduler service."""
    print(f"Scheduler initialized at {datetime.now().isoformat()}")
    print(f"Interval: {SCHEDULER_INTERVAL_DAYS} days")

    run_startup_bootstrap_if_needed()

    scheduler = BackgroundScheduler()
    scheduler.add_job(
        run_pipeline,
        trigger=IntervalTrigger(days=SCHEDULER_INTERVAL_DAYS),
        name="data_update",
        replace_existing=True,
        id="data_update_job",
    )

    scheduler.start()

    try:
        print("Scheduler running and waiting for recurring jobs")
        while True:
            time.sleep(60)
    except KeyboardInterrupt:
        print("Scheduler stopped")
        scheduler.shutdown()


if __name__ == "__main__":
    main()
