"""Modular experiment queue runner.

Reads a batch TOML that defines which experiments to run and in what order.
Each experiment runs as a fresh subprocess, so code changes between runs
take effect automatically. Kill anytime, restart, picks up where it left off.

Batch TOML format::

    [batch]
    name = "Phase 1 Re-Run"
    clean = true                          # delete results before each run
    experiments = ["quick", "full", ...]  # sequential execution order
    then = ["sweep_streams_8"]            # run after all above complete

Each entry in ``experiments`` maps to ``experiments/{name}.toml``.

Completion check: an experiment is "done" if its ``results/{name}/config.json``
contains a ``completed_at`` timestamp.

Notifications: sends macOS notifications on experiment completion and writes
a signal file (``.claude/queue-status.json``) for tooling integration.

Usage:
    uv run python -m scripts.run_queue experiments/batches/phase1_rerun.toml
    uv run python -m scripts.run_queue experiments/batches/phase1_rerun.toml --dry-run
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
import tomllib
from datetime import UTC, datetime
from pathlib import Path

EXPERIMENTS_DIR = Path("experiments")
RESULTS_DIR = Path("results")
SIGNAL_FILE = Path(".claude/queue-status.json")


def _notify(title: str, message: str) -> None:
    """Send a macOS notification."""
    subprocess.run(
        [
            "osascript", "-e",
            f'display notification "{message}" with title "{title}"',
        ],
        check=False,
        capture_output=True,
    )


def _write_signal(
    batch_name: str,
    experiment: str,
    status: str,
    done: int,
    total: int,
) -> None:
    """Write a signal file for external tooling (Claude Code, scripts).

    The file is overwritten on every event so readers just need to
    check mtime + contents.
    """
    SIGNAL_FILE.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "batch": batch_name,
        "experiment": experiment,
        "status": status,
        "done": done,
        "total": total,
        "timestamp": datetime.now(UTC).isoformat(),
    }
    SIGNAL_FILE.write_text(json.dumps(payload, indent=2))


def _is_complete(run_name: str) -> bool:
    """Check if an experiment has completed via results config.

    An experiment is done if ``results/{name}/config.json`` contains
    a ``completed_at`` timestamp, set by the experiment runner on success.
    This is the ground truth — TOML meta.status can be stale or wrong.
    """
    config_path = RESULTS_DIR / run_name / "config.json"
    if not config_path.exists():
        return False
    try:
        with open(config_path) as f:
            data = json.load(f)
        return bool(data.get("completed_at"))
    except (json.JSONDecodeError, OSError):
        return False


def _run_experiment(name: str, clean: bool = False) -> bool:
    """Run a single experiment as a subprocess.

    Args:
        name: Experiment name (TOML filename stem).
        clean: If True, delete results directory before running.

    Returns:
        True if the experiment completed successfully.
    """
    toml_path = EXPERIMENTS_DIR / f"{name}.toml"
    if not toml_path.exists():
        print(f"  ERROR: {toml_path} not found, skipping")
        return False

    if clean:
        results_dir = RESULTS_DIR / name
        if results_dir.exists():
            import shutil

            shutil.rmtree(results_dir)
            print(f"  Cleaned {results_dir}")

    result = subprocess.run(
        ["uv", "run", "python", "-m", "scripts.experiment", str(toml_path)],
        check=False,
    )
    return result.returncode == 0


def _run_batch(
    names: list[str],
    clean: bool,
    dry_run: bool,
    batch_name: str = "",
    label: str = "",
    total_in_batch: int = 0,
) -> None:
    """Run a list of experiments sequentially, skipping completed ones.

    Args:
        names: Experiment names in execution order.
        clean: Delete results before each run.
        dry_run: Just print, don't run.
        batch_name: Batch name for notifications.
        label: Label for this batch phase.
        total_in_batch: Total experiments across all phases (for signal file).
    """
    done = [n for n in names if _is_complete(n)]
    pending = [n for n in names if not _is_complete(n)]

    if label:
        print(f"\n--- {label} ---")
    print(f"  Total: {len(names)}, Done: {len(done)}, Pending: {len(pending)}")

    for n in names:
        marker = "  +" if _is_complete(n) else "  o"
        print(f"{marker} {n}")

    if dry_run or not pending:
        return

    for i, name in enumerate(pending):
        ts = datetime.now(UTC).strftime("%H:%M:%S")
        print(f"\n[{i + 1}/{len(pending)}] {name} ({ts})")
        print("-" * 40)

        _write_signal(batch_name, name, "running", 0, total_in_batch)

        t0 = time.monotonic()
        ok = _run_experiment(name, clean=clean)
        elapsed = time.monotonic() - t0

        if ok:
            print(f"  Done: {name} ({elapsed / 60:.1f} min)")
            _notify("KromCanon", f"{name} done ({elapsed / 60:.1f} min)")
            _write_signal(batch_name, name, "done", i + 1 + len(done), total_in_batch)
        else:
            print(f"  FAILED: {name} ({elapsed / 60:.1f} min)")
            _notify("KromCanon", f"{name} FAILED ({elapsed / 60:.1f} min)")
            _write_signal(batch_name, name, "failed", i + len(done), total_in_batch)


def main() -> None:
    """Main entry point."""
    dry_run = "--dry-run" in sys.argv
    args = [a for a in sys.argv[1:] if not a.startswith("--")]

    if not args:
        print(
            "Usage: python -m scripts.run_queue "
            "<batch.toml> [--dry-run]"
        )
        sys.exit(1)

    batch_path = Path(args[0])
    if not batch_path.exists():
        print(f"Error: {batch_path} not found")
        sys.exit(1)

    with open(batch_path, "rb") as f:
        raw = tomllib.load(f)

    batch = raw.get("batch", {})
    name = batch.get("name", batch_path.stem)
    clean = batch.get("clean", False)
    experiments = batch.get("experiments", [])
    then = batch.get("then", [])

    print("=" * 60)
    print(f"BATCH: {name}")
    print(f"Clean: {clean}")
    ts = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    print(f"Time: {ts}")
    print("=" * 60)

    total = len(experiments) + len(then)

    _run_batch(
        experiments, clean, dry_run,
        batch_name=name, label="Main", total_in_batch=total,
    )

    if then:
        if dry_run:
            _run_batch(
                then, clean, dry_run,
                batch_name=name, label="Then", total_in_batch=total,
            )
        else:
            # Only start "then" if all main experiments complete
            all_done = all(_is_complete(n) for n in experiments)
            if all_done:
                _run_batch(
                    then, clean, dry_run,
                    batch_name=name, label="Then", total_in_batch=total,
                )
            else:
                failed = [n for n in experiments if not _is_complete(n)]
                print(f"\nSkipping 'then' phase: {len(failed)} "
                      f"main experiments incomplete")

    print("\n" + "=" * 60)
    all_names = experiments + then
    done_count = sum(1 for n in all_names if _is_complete(n))
    print(f"BATCH COMPLETE: {done_count}/{len(all_names)} done")
    print("=" * 60)

    _notify("KromCanon", f"Batch complete: {done_count}/{len(all_names)} done")
    _write_signal(name, "BATCH", "complete", done_count, len(all_names))


if __name__ == "__main__":
    main()
