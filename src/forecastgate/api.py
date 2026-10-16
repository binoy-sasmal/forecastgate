"""Read-only HTTP API over the committed prediction artifact.

Every request re-validates the artifact (see serving.py), so a file changed after start-up
is refused on the next request. A refusal is a 503 carrying the reason; it is never
answered with an older artifact.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse

from forecastgate.serving import ServedArtifact, ServingRefusedError, load_current


def create_app(predictions_dir: Path, *, as_of: date, max_age_days: int) -> FastAPI:
    app = FastAPI(title="forecastgate", description="Read-only weekly sales forecasts.")

    def current() -> ServedArtifact:
        try:
            return load_current(predictions_dir, as_of=as_of, max_age_days=max_age_days)
        except ServingRefusedError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

    @app.get("/health")
    def health() -> JSONResponse:
        try:
            served = load_current(predictions_dir, as_of=as_of, max_age_days=max_age_days)
        except ServingRefusedError as exc:
            return JSONResponse(
                status_code=503,
                content={"status": "unavailable", "reason": str(exc), "as_of": f"{as_of}"},
            )
        return JSONResponse(
            content={
                "status": "ok",
                "identity": served.identity,
                "model_version": served.manifest["identity"]["model_version"],
                "cutoff_week": f"{served.cutoff_week}",
                "data_through": f"{served.data_through}",
                "age_days": served.age_days,
                "as_of": f"{as_of}",
                "products": int(served.predictions["stock_code"].nunique()),
            }
        )

    @app.get("/predictions/{stock_code}")
    def predictions(stock_code: str) -> dict[str, Any]:
        served = current()
        rows = served.predictions[served.predictions["stock_code"] == stock_code]
        if rows.empty:
            raise HTTPException(status_code=404, detail=f"{stock_code} is not in the forecast")
        ordered = rows.sort_values("horizon")
        return {
            "stock_code": stock_code,
            "identity": served.identity,
            "model_version": served.manifest["identity"]["model_version"],
            "cutoff_week": f"{served.cutoff_week}",
            "forecasts": [
                {
                    "horizon": int(horizon),
                    "target_week": f"{target_week:%Y-%m-%d}",
                    "prediction": float(prediction),
                }
                for horizon, target_week, prediction in zip(
                    ordered["horizon"], ordered["target_week"], ordered["prediction"], strict=True
                )
            ],
        }

    return app
