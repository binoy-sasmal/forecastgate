# One image for every entrypoint. Same commands as on a laptop and in CI:
#   docker run --rm --network none forecastgate uv run forecastgate reproduce --fixture
#   docker run --rm forecastgate uv run forecastgate reproduce      # full data, downloads
#
# The image carries git metadata next to the working tree, so the gate freeze is verified
# inside the container exactly as in a checkout, and provenance records the real commit
# and whether the tree the image was built from had uncommitted changes.

# --- Stage 1: assemble a sanitised source tree -------------------------------------
# Only this stage ever sees the build context's .git. It keeps the history needed for
# the freeze check (HEAD, objects, refs, packed-refs, index) and writes a fresh git
# config, so the original config (remote URLs, credential helpers, extra headers), the
# reflogs and the hooks never reach the final image. This stage's layers are discarded.
FROM python:3.12-slim AS source
RUN apt-get update \
    && apt-get install --yes --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*
COPY . /src
RUN set -eu; \
    mkdir -p /out/.git; \
    cd /src; \
    tar --exclude=./.git -cf - . | tar -xf - -C /out; \
    cp -r .git/HEAD .git/objects .git/refs .git/index /out/.git/; \
    if [ -f .git/packed-refs ]; then cp .git/packed-refs /out/.git/; fi; \
    git -C /out init --quiet; \
    # Compare content, not permission bits: a Windows build context does not carry the
    # file modes git recorded, which would make every tracked file look modified.
    git -C /out config core.filemode false; \
    rm -rf /out/.git/hooks; \
    git -C /out rev-parse --verify HEAD > /dev/null

# --- Stage 2: the runtime image -----------------------------------------------------
FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:0.13.1 /uv /uvx /bin/

ENV UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    UV_PYTHON_DOWNLOADS=never \
    UV_NO_SYNC=1

WORKDIR /app
# libgomp1: LightGBM links against the OpenMP runtime. git: verifies the gate freeze.
RUN apt-get update \
    && apt-get install --yes --no-install-recommends libgomp1 git \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 10001 app

# Dependencies first, so source edits do not invalidate this layer.
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --locked --no-dev --no-install-project

COPY --from=source --chown=app:app /out /app
RUN uv sync --locked --no-dev && chown app /app

USER app
CMD ["uv", "run", "forecastgate", "reproduce", "--fixture"]
