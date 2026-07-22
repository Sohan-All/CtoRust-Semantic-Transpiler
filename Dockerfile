# diffusionMTUs trial sandbox.
#
# The image holds only the toolchain (Python deps + Rust for the compile
# loop); the repo itself is bind-mounted READ-ONLY at /pipeline by
# docker_trial.sh, so a run can never modify pipeline code, and all state
# and artifacts go to the /out volume (DIFFUSIONMTUS_OUT).
#
#   docker build -t diffusionmtus .
#   ./docker_trial.sh <c_src_dir> [run_project.py args...]
FROM rust:1.88-slim-bookworm

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-pip python3-venv \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

RUN python3 -m venv /opt/venv \
    && /opt/venv/bin/pip install --no-cache-dir \
        openai httpx pydantic tqdm tree-sitter tree-sitter-c

ENV PATH=/opt/venv/bin:$PATH \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    # everything a run writes lands under the /out volume
    DIFFUSIONMTUS_OUT=/out \
    CARGO_HOME=/out/.cargo \
    HOME=/out

WORKDIR /pipeline
ENTRYPOINT ["python3", "run_project.py"]
