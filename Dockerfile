# One image for every entrypoint. Same commands as on a laptop and in CI:
#   docker run --rm --network none forecastgate uv run forecastgate reproduce --fixture
#   docker run --rm forecastgate uv run forecastgate reproduce      # full data, downloads
#
# The image carries the repository's git metadata (.git) next to the working tree, so the
# gate freeze is verified inside the container exactly as in a checkout: gate.yaml must be
# tracked and identical to its committed version, and provenance records the real commit
# and whether the tree the image was built from had uncommitted changes.
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

# The whole checkout, including .git; .dockerignore keeps out downloaded data, caches,
# local registries, artifacts and credential files. Remotes and any credential settings
# are dropped from the copied git config, so a token in a remote URL never ships; the
# freeze check needs only local history.
COPY . .
RUN for remote in $(git remote); do git remote remove "$remote"; done \
    && (git config --local --remove-section credential 2>/dev/null || true)
RUN uv sync --locked --no-dev && chown -R app /app

USER app
CMD ["uv", "run", "forecastgate", "reproduce", "--fixture"]
