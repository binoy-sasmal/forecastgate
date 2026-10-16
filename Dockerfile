# One image for every entrypoint. Same command as on a laptop and in CI:
#   docker run --rm --network none forecastgate uv run forecastgate reproduce --fixture
FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:0.13.1 /uv /uvx /bin/

ENV UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    UV_PYTHON_DOWNLOADS=never \
    UV_NO_SYNC=1

WORKDIR /app
RUN useradd --create-home --uid 10001 app

# Dependencies first, so source edits do not invalidate this layer.
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --locked --no-dev --no-install-project

COPY src ./src
COPY data/fixture ./data/fixture
RUN uv sync --locked --no-dev && chown -R app /app

ARG GIT_SHA=unknown
ENV FORECASTGATE_GIT_SHA=${GIT_SHA}

USER app
CMD ["uv", "run", "forecastgate", "reproduce", "--fixture"]
