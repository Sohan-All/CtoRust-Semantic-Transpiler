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

# Install from the manifest rather than an inline list, so the image and a
# local venv cannot drift. The previous inline list omitted `anthropic`, which
# llm.py needs for the Vertex path, and carried `tqdm`, which nothing imports.
# Only the manifest is COPYed — the repo arrives later as a read-only bind
# mount, and keeping it out of this layer preserves pip's build cache.
COPY requirements.txt /tmp/requirements.txt
RUN python3 -m venv /opt/venv \
    && /opt/venv/bin/pip install --no-cache-dir -r /tmp/requirements.txt \
    && rm /tmp/requirements.txt

ENV PATH=/opt/venv/bin:$PATH \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    # everything a run writes lands under the /out volume
    DIFFUSIONMTUS_OUT=/out \
    CARGO_HOME=/out/.cargo \
    HOME=/out

WORKDIR /pipeline
ENTRYPOINT ["python3", "run_project.py"]
