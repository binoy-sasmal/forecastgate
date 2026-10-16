"""Run every failure demonstration and write docs/failures.md from the captured output.

Usage: uv run python scripts/failure_demos.py [--docker]

Each demonstration runs in a temporary `git clone` of this repository, so destructive
steps (corrupting the archive, editing gate.yaml, tampering with artifacts) never touch
the working tree. Every command's exit code is checked against the expected one; any
mismatch stops the script without writing the document. Commands run through the
project's entry point (`forecastgate.cli:main`, which `uv run forecastgate` invokes) and
pytest. Full-data demonstrations need the downloaded archive in data/raw; `--docker`
adds the image-layer credential scan, which builds two images.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
OUTPUT = REPO / "docs" / "failures.md"
RAW = REPO / "data" / "raw" / "online_retail_ii.zip"
CACHE = REPO / "artifacts" / "full" / "cache"
CLI = ["-c", "from forecastgate.cli import main; main()"]


@dataclass
class Demo:
    title: str
    shows: str
    evidence: str
    blocks: list[str] = field(default_factory=list)


class Runner:
    def __init__(self, tmp: Path) -> None:
        self.tmp = tmp
        self.env = {k: v for k, v in os.environ.items() if k != "FORECASTGATE_GIT_SHA"}
        self.env.update({"PYTHONIOENCODING": "utf-8", "COLUMNS": "100", "NO_COLOR": "1"})

    def clean(self, text: str) -> str:
        for path in (str(self.tmp), str(REPO)):
            for variant in (path, path.replace("\\", "/")):
                text = text.replace(variant, "<tmp>" if path == str(self.tmp) else "<repo>")
        text = re.sub(r"<(tmp|repo)>[^\s,)]*", lambda m: m.group(0).replace("\\", "/"), text)
        text = re.sub(r" in \d+\.\d+s( \([0-9:]+\))?", "", text)  # pytest timings
        return text.rstrip()

    def clone(self, name: str) -> Path:
        target = self.tmp / name
        subprocess.run(["git", "clone", "-q", str(REPO), str(target)], check=True)
        return target

    def command(
        self, demo: Demo, shown: str, argv: list[str], cwd: Path, expect: int, *, keep: int = 0
    ) -> str:
        result = subprocess.run(
            argv, cwd=cwd, env=self.env, capture_output=True, text=True, encoding="utf-8"
        )
        output = self.clean(result.stdout + result.stderr)
        if result.returncode != expect:
            raise SystemExit(
                f"{demo.title}: `{shown}` exited {result.returncode}, expected {expect}\n{output}"
            )
        lines = output.splitlines()
        if keep and len(lines) > keep:
            lines = [*lines[:keep], f"... ({len(lines) - keep} more lines)"]
        body = "\n".join(lines)
        demo.blocks.append(
            f"$ {shown}\n{body}\n[exit {result.returncode}]".replace("\n\n[exit", "\n[exit")
        )
        return output

    def cli(self, demo: Demo, args: list[str], cwd: Path, expect: int, **kw: int) -> str:
        return self.command(
            demo,
            "uv run forecastgate " + " ".join(args),
            [sys.executable, *CLI, *args],
            cwd,
            expect,
            **kw,
        )

    def pytest(self, demo: Demo, target: str, select: str, expect: int = 0) -> str:
        argv = [
            sys.executable,
            "-m",
            "pytest",
            target,
            "-k",
            select,
            "-v",
            "-p",
            "no:cacheprovider",
            "--no-header",
        ]
        result = subprocess.run(
            argv, cwd=REPO, env=self.env, capture_output=True, text=True, encoding="utf-8"
        )
        if result.returncode != expect:
            raise SystemExit(
                f"{demo.title}: pytest {target} -k {select} exited {result.returncode}\n"
                + result.stdout
            )
        keep = [
            line
            for line in result.stdout.splitlines()
            if "PASSED" in line or "FAILED" in line or re.search(r"\d+ passed", line)
        ]
        keep = [re.sub(r"\s+\[\s*\d+%\]$", "", line) for line in keep]
        demo.blocks.append(
            f'$ uv run pytest {target} -k "{select}"\n'
            + self.clean("\n".join(keep))
            + f"\n[exit {result.returncode}]"
        )
        return result.stdout

    def step(self, demo: Demo, description: str, action: Callable[[], str | None]) -> None:
        note = action()
        demo.blocks.append(f"# {description}" + (f"\n{note}" if note else ""))


def flip_byte(path: Path) -> None:
    payload = bytearray(path.read_bytes())
    payload[len(payload) // 2] ^= 0xFF
    path.write_bytes(bytes(payload))


def pooled_wape(report: Path, recipe: str) -> str:
    for line in report.read_text(encoding="utf-8").splitlines():
        if line.startswith(f"| `{recipe}` |") and line.count("|") == 6:
            return line
    raise SystemExit(f"no pooled row for {recipe} in {report}")


def demos(run: Runner, *, docker: bool) -> list[Demo]:
    out: list[Demo] = []
    full = RAW.exists()

    # 1 -------------------------------------------------------------------------------
    d = Demo(
        "Invalid input is rejected before anything is written",
        "A corrupted archive fails the pinned SHA-256; archives with a correct hash but bad "
        "content fail the raw data contract or the sheet-overlap check.",
        "`tests/test_reproduce_fixture.py`, `tests/test_audit_and_cache.py`, "
        "`tests/test_ingest.py`",
    )
    repo = run.clone("invalid-input")
    run.step(
        d,
        "flip one byte in the middle of data/fixture/online_retail_II_fixture.zip",
        lambda: flip_byte(repo / "data/fixture/online_retail_II_fixture.zip"),
    )
    run.cli(d, ["reproduce", "--fixture"], repo, 3)
    run.step(
        d,
        "is there an artifacts directory?",
        lambda: "yes" if (repo / "artifacts").exists() else "no: nothing was written",
    )
    run.pytest(d, "tests/test_audit_and_cache.py", "contract_rejects_modified_archive")
    out.append(d)

    # 2 -------------------------------------------------------------------------------
    d = Demo(
        "Leakage is detected, including two deliberately planted leaks",
        "Features built from data truncated at a point in time must equal features built "
        "from full data whose later records were altered; every training label must satisfy "
        "t + h <= T. `future_window` (a 4-week mean reading y[t+1]) is caught by truncation "
        "equivalence; `origin_filter` (labels filtered on the origin only) by label "
        "eligibility. In the backtest harness, recipes only receive panels truncated at the "
        "fold cutoff, so `origin_filter` cannot reach future labels there at all, while "
        "`future_window` still changes the metrics.",
        "`tests/test_leakage.py`, `tests/test_features.py`",
    )
    run.pytest(d, "tests/test_leakage.py", "truncation or planted or labels")
    repo = run.clone("leakage")
    run.cli(d, ["backtest", "--fixture"], repo, 0)
    run.cli(d, ["backtest", "--fixture", "--inject-leak", "future_window"], repo, 0, keep=2)
    run.cli(d, ["backtest", "--fixture", "--inject-leak", "origin_filter"], repo, 0, keep=2)
    reports = repo / "artifacts/fixture/reports"
    run.step(
        d,
        "pooled lgbm row (recipe | WAPE | bias | MASE | excluded) in each report",
        lambda: "\n".join(
            f"{name}: {pooled_wape(reports / name, 'lgbm')}"
            for name in (
                "backtest.md",
                "backtest_leak_future_window.md",
                "backtest_leak_origin_filter.md",
            )
        ),
    )
    out.append(d)

    # 3 -------------------------------------------------------------------------------
    d = Demo(
        "The gate blocks a model that does not meet the frozen criteria",
        "`gate` exits 10 on a quality rejection and leaves the champion alias unchanged; on "
        "an empty registry it registers the best single baseline. Publication continues with "
        "the champion. On the full data, lgbm fails criteria 1, 2 and 4.",
        "`tests/test_registry.py`, `tests/test_gate.py`, "
        "[reports/backtest.md](../reports/backtest.md)",
    )
    repo = run.clone("gate")
    run.cli(d, ["predict", "--fixture"], repo, 9)
    run.cli(d, ["gate", "--fixture"], repo, 10)
    run.cli(d, ["gate", "--fixture"], repo, 10)
    run.step(
        d,
        "registry: versions and alias",
        lambda: json.dumps(
            {
                "aliases": (
                    data := json.loads(
                        (repo / "artifacts/fixture/registry/registry.json").read_text()
                    )
                )["aliases"],
                "versions": [
                    f"{v['recipe']}@v{v['version']}: {v['reason']}" for v in data["versions"]
                ],
            },
            indent=2,
        ),
    )
    run.cli(d, ["predict", "--fixture"], repo, 0)
    run.pytest(d, "tests/test_registry.py", "weak_candidate or passing_candidate or stale_champion")
    if full:
        (repo / "data/raw").mkdir(parents=True, exist_ok=True)
        shutil.copy(RAW, repo / "data/raw/online_retail_ii.zip")
        if CACHE.exists():
            shutil.copytree(CACHE, repo / "artifacts/full/cache")
        run.cli(d, ["gate"], repo, 10)
    out.append(d)

    # 4 -------------------------------------------------------------------------------
    d = Demo(
        "Full-data evaluation only runs against the committed, unchanged gate",
        "Before any download, a full-data run requires gate.yaml to be tracked and identical "
        "to its committed version; the effective configuration must also equal it.",
        "`tests/test_gate.py`",
    )
    repo = run.clone("freeze")
    gate = repo / "gate.yaml"

    def relax() -> None:
        text = gate.read_text(encoding="utf-8")
        gate.write_text(text.replace("abs_bias_max: 0.05", "abs_bias_max: 0.50"), encoding="utf-8")

    run.step(d, "edit gate.yaml: abs_bias_max 0.05 -> 0.50 (relaxing the bias criterion)", relax)
    run.cli(d, ["backtest"], repo, 7)
    run.cli(d, ["gate"], repo, 7)
    loose = run.tmp / "not-a-checkout"
    shutil.copytree(REPO / "data/fixture", loose / "data/fixture")
    shutil.copy(REPO / "gate.yaml", loose / "gate.yaml")
    run.step(d, "a directory with gate.yaml but no git history", lambda: None)
    run.cli(d, ["backtest"], loose, 7)
    run.pytest(d, "tests/test_gate.py", "freeze or frozen or mismatch")
    out.append(d)

    # 5 -------------------------------------------------------------------------------
    d = Demo(
        "Publication is atomic, idempotent and refuses inconsistent reruns",
        "A rerun with the same identity and content is a no-op; the same identity with "
        "different content is refused. Contract failures and interrupted runs never change "
        "CURRENT.",
        "`tests/test_publish.py`, `tests/test_reproduce_fixture.py`",
    )
    repo = run.clone("publication")
    run.cli(d, ["reproduce", "--fixture"], repo, 0, keep=3)
    run.cli(d, ["reproduce", "--fixture"], repo, 0)
    predictions = repo / "artifacts/fixture/predictions"

    def tamper() -> str:
        target = predictions / (predictions / "CURRENT").read_text().strip() / "predictions.csv"
        lines = target.read_text(encoding="utf-8").splitlines()
        code, cutoff, horizon, week, _ = lines[1].split(",")
        lines[1] = f"{code},{cutoff},{horizon},{week},999.000000"
        target.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return f"first row is now: {lines[1]}"

    run.step(d, "overwrite one forecast in the published predictions.csv", tamper)
    run.cli(d, ["reproduce", "--fixture"], repo, 5)
    run.pytest(
        d,
        "tests/test_publish.py",
        "contract_failure or crash_before_pointer or held_lock or new_model_version",
    )
    out.append(d)

    # 6 -------------------------------------------------------------------------------
    d = Demo(
        "Serving refuses stale, future-dated and tampered artifacts",
        "Serving re-validates on every request and checks freshness against an explicit "
        "simulated date: data through the end of the cutoff week must be 0 to 7 days old. "
        "The fixture artifact covers data through 2011-12-04. A refusal is HTTP 503 with the "
        "reason (`serve-check` exits 11); there is no fallback to an older artifact.",
        "`tests/test_serving.py`",
    )
    repo = run.clone("serving")
    run.cli(d, ["reproduce", "--fixture"], repo, 0, keep=3)
    run.cli(d, ["serve-check", "--fixture", "--as-of", "2011-12-05"], repo, 0)
    run.cli(d, ["serve-check", "--fixture", "--as-of", "2011-12-12"], repo, 11)
    run.cli(d, ["serve-check", "--fixture", "--as-of", "2011-12-03"], repo, 11)
    predictions = repo / "artifacts/fixture/predictions"

    def tamper_served() -> None:
        target = predictions / (predictions / "CURRENT").read_text().strip() / "predictions.csv"
        target.write_bytes(target.read_bytes().replace(b".", b"9", 1))

    run.step(d, "change one byte of the served predictions.csv", tamper_served)
    run.cli(d, ["serve-check", "--fixture", "--as-of", "2011-12-05"], repo, 11)
    run.pytest(
        d,
        "tests/test_serving.py",
        "freshness or tampering or another_artifact or failed_publication",
    )
    out.append(d)

    # 7 -------------------------------------------------------------------------------
    if docker:
        d = Demo(
            "Git credentials reach no layer of the image",
            "The image carries git metadata for the freeze check. A discarded build stage "
            "reads the context's .git and passes on only history and a fresh config. Fake "
            "credentials are planted in a clone's .git/config, then every decompressed layer "
            "is searched; the discarded stage is the positive control.",
            "`Dockerfile`, `scripts/scan_image_layers.py`",
        )
        repo = run.clone("docker")
        # A fresh random marker per run: it exists nowhere except the planted config, so
        # a hit can only come from the credentials (this script's own source cannot match).
        marker = f"FAKE-LEAK-{uuid.uuid4().hex}"
        header = f"AUTHORIZATION: basic {marker}-1"
        url = f"https://{marker}-2@example.invalid/r.git"
        run.command(
            d,
            f'git config --local http.extraheader "{header}"',
            ["git", "config", "--local", "http.extraheader", header],
            repo,
            0,
        )
        run.command(
            d, f"git remote add leaktest {url}", ["git", "remote", "add", "leaktest", url], repo, 0
        )
        images = {"final": "forecastgate:leak-demo", "source": "forecastgate:leak-demo-source"}
        run.command(
            d,
            f"docker build -q -t {images['final']} .",
            ["docker", "build", "-q", "-t", images["final"], "."],
            repo,
            0,
        )
        run.command(
            d,
            f"docker build -q --target source -t {images['source']} .",
            ["docker", "build", "-q", "--target", "source", "-t", images["source"], "."],
            repo,
            0,
        )
        scan = [sys.executable, str(REPO / "scripts/scan_image_layers.py")]
        hits = {}
        for stage, image in images.items():
            output = run.command(
                d,
                f"uv run python scripts/scan_image_layers.py {image} {marker}",
                [*scan, image, marker],
                repo,
                0,
            )
            found = re.search(r"(\d+) occurrences", output)
            hits[stage] = int(found.group(1)) if found else -1
        subprocess.run(["docker", "rmi", "-f", *images.values()], capture_output=True, check=False)
        if hits["final"] != 0 or hits["source"] < 1:
            raise SystemExit(f"credential scan failed: {hits}")
        for i, block in enumerate(d.blocks):
            block = re.sub(r"sha256:[0-9a-f]{64}", "sha256:<image id>", block)
            d.blocks[i] = block.replace(marker, "FAKE-LEAK-<random>")
        d.blocks.append(
            f"# checked: final image {hits['final']} occurrences (must be 0); "
            f"discarded source stage {hits['source']} (positive control, must be at least 1)"
        )
        out.append(d)
    return out


def render(items: list[Demo], *, commit: str, dirty: bool, docker: bool, full: bool) -> str:
    lines = [
        "# Failure demonstrations",
        "",
        "Generated by `uv run python scripts/failure_demos.py"
        + (" --docker" if docker else "")
        + "`",
        f"at commit `{commit}`" + (" with uncommitted changes." if dirty else " (clean tree)."),
        "Every command below was run, and its exit code checked against the expected one, in a",
        "temporary clone of the repository; `<tmp>` stands for that directory. Commands run",
        "through the project's entry point (`forecastgate.cli:main`, which `uv run forecastgate`",
        "invokes).",
        "",
        "Exit codes: 0 success; 3 data contract; 4 prediction contract; 5 publication",
        "refused; 6 publication lock held; 7 gate not frozen; 8 gate config mismatch; 9",
        "registry (no or stale champion); 10 gate rejected the candidate; 11 serving refused.",
        "",
        "These demonstrations are local runs. GitHub Actions CI status, with links to runs,",
        "is recorded in the README.",
    ]
    if not full:
        lines += ["", "Full-data steps were skipped: the downloaded archive was not present."]
    for number, demo in enumerate(items, 1):
        lines += ["", f"## {number}. {demo.title}", "", demo.shows, "", f"Tests: {demo.evidence}"]
        for block in demo.blocks:
            lines += ["", "```", block, "```"]
    lines.append("")
    return "\n".join(lines)


def main(argv: list[str]) -> int:
    docker = "--docker" in argv[1:]
    commit = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"],
        cwd=REPO,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    dirty = bool(
        subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            cwd=REPO,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    )
    with tempfile.TemporaryDirectory(prefix="forecastgate-demos-") as tmp:
        items = demos(Runner(Path(tmp).resolve()), docker=docker)
    OUTPUT.write_bytes(
        render(items, commit=commit, dirty=dirty, docker=docker, full=RAW.exists()).encode("utf-8")
    )
    print(f"wrote {OUTPUT.relative_to(REPO)} with {len(items)} demonstrations")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
