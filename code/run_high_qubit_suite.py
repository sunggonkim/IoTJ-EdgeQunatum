#!/usr/bin/env python3
import argparse
import datetime as _dt
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import List, Optional

DEFAULT_CIRCUITS = ["QV", "VQC", "QSVM", "Random", "GHZ", "VQE"]
DEFAULT_DEPTH = 5
DEFAULT_SIM_MODE = "async"  # tiered pipeline (legacy name)


def _now_ts() -> str:
    return _dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def log(msg: str) -> None:
    print(f"[{_now_ts()}] {msg}", flush=True)


def load_results(path: Path) -> dict:
    if path.exists():
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    return {"runs": [], "meta": {"start_time": _dt.datetime.now().isoformat()}}


def save_results(path: Path, results: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    tmp.replace(path)


def already_run(results: dict, qubits: int, circuit: str, depth: int) -> bool:
    for r in results.get("runs", []):
        if (
            r.get("scheme") == "EdgeQuantum"
            and r.get("qubits") == qubits
            and r.get("circuit") == circuit
            and r.get("depth", depth) == depth
            and r.get("success") is True
        ):
            return True
    return False


def parse_total_time(stdout_text: str) -> Optional[float]:
    for line in stdout_text.splitlines():
        if "Total Time:" in line:
            try:
                return float(line.split(":", 1)[1].strip().split()[0])
            except Exception:
                continue
    return None


def disk_free_bytes(path: Path) -> int:
    usage = shutil.disk_usage(path)
    return int(usage.free)


def predicted_needed_bytes(qubits: int, assumed_ratio: float) -> int:
    # raw state size = 2^q * sizeof(cuComplex) = 2^q * 8 bytes
    raw = (1 << qubits) * 8
    # ping-pong keeps 2 files around
    return int(2 * raw * assumed_ratio)


def try_jetson_clocks() -> None:
    # Repo instruction: system password 1234qwer
    cmd = "echo '1234qwer' | sudo -S /usr/bin/jetson_clocks 2>/dev/null"
    os.system(cmd)


def run_one(
    binary: Path,
    qubits: int,
    circuit: str,
    depth: int,
    storage_path: Path,
    results_path: Path,
    log_dir: Path,
    sim_mode: str,
    force_mode: bool,
    cleanup_on_success: bool,
    keep_storage_on_failure: bool,
    rerun: bool,
) -> None:
    results = load_results(results_path)

    if (not rerun) and already_run(results, qubits, circuit, depth):
        log(f"[SKIP] EdgeQuantum | {circuit} | {qubits}q (already in results)")
        return

    log_dir.mkdir(parents=True, exist_ok=True)
    run_log = log_dir / f"edgeq_{qubits}q_{circuit}_d{depth}_{_dt.datetime.now().strftime('%Y%m%dT%H%M%S')}.log"

    cmd = [
        str(binary),
        "--qubits",
        str(qubits),
        "--circuit",
        circuit,
        "--depth",
        str(depth),
        "--storage",
        str(storage_path),
        "--sim-mode",
        sim_mode,
    ]
    if force_mode:
        cmd.append("--force-mode")

    log(f"[RUN] EdgeQuantum | {circuit} | {qubits}q | log={run_log}")
    log(f"  CMD: {' '.join(cmd)}")

    start = time.time()
    stdout_buf: List[str] = []
    success = False
    error: Optional[str] = None

    # Stream stdout to both console and per-run log.
    with run_log.open("w", encoding="utf-8") as lf:
        lf.write(f"# started: {_dt.datetime.now().isoformat()}\n")
        lf.write(f"# cmd: {' '.join(cmd)}\n\n")
        lf.flush()

        p = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True,
        )

        assert p.stdout is not None
        for line in p.stdout:
            lf.write(line)
            lf.flush()
            sys.stdout.write(line)
            sys.stdout.flush()
            # keep last ~400 lines for JSON error context
            stdout_buf.append(line)
            if len(stdout_buf) > 400:
                stdout_buf = stdout_buf[-400:]

        rc = p.wait()

    wall = time.time() - start

    if rc == 0:
        success = True
    else:
        error = f"Exit Code {rc}"

    stdout_tail = "".join(stdout_buf)
    sim_time = parse_total_time(stdout_tail) if success else None

    # Record result
    entry = {
        "scheme": "EdgeQuantum",
        "type": "cpp",
        "mode": sim_mode,
        "force_mode": bool(force_mode),
        "qubits": int(qubits),
        "circuit": circuit,
        "depth": int(depth),
        "success": bool(success),
        "sim_time": float(sim_time) if sim_time is not None else None,
        "wall_time": float(wall),
        "timestamp": _dt.datetime.now().isoformat(),
        "log_file": str(run_log.relative_to(results_path.parent)),
    }

    if not success:
        entry["error"] = error
        entry["logs"] = stdout_tail[-4000:]

    results.setdefault("runs", []).append(entry)
    results.setdefault("meta", {})["last_update"] = _dt.datetime.now().isoformat()
    save_results(results_path, results)

    # Cleanup storage artifacts so we don't fill disk across runs
    if success and cleanup_on_success:
        patterns = [
            str(storage_path) + "*",
            str(storage_path) + ".lz4*",
        ]
        removed = 0
        for pat in patterns:
            for pth in storage_path.parent.glob(Path(pat).name):
                try:
                    pth.unlink()
                    removed += 1
                except Exception:
                    pass
        log(f"[CLEANUP] removed {removed} files under {storage_path.parent}")
    elif (not success) and (not keep_storage_on_failure):
        for pth in storage_path.parent.glob(storage_path.name + "*"):
            try:
                pth.unlink()
            except Exception:
                pass


def main() -> int:
    script_dir = Path(__file__).resolve().parent
    default_binary = script_dir / "build" / "edge_quantum"

    ap = argparse.ArgumentParser()
    ap.add_argument("--qubits", type=int, nargs="+", required=True)
    ap.add_argument("--circuits", type=str, nargs="+", default=DEFAULT_CIRCUITS)
    ap.add_argument("--depth", type=int, default=DEFAULT_DEPTH)
    ap.add_argument("--sim-mode", type=str, default=DEFAULT_SIM_MODE)
    ap.add_argument("--force-mode", action="store_true", default=True)
    ap.add_argument("--no-force-mode", action="store_true")
    ap.add_argument("--binary", type=str, default=str(default_binary))
    ap.add_argument("--results-file", type=str, default=str(script_dir / "comprehensive_results.json"))
    ap.add_argument("--log-dir", type=str, default=str(script_dir / "run_logs" / "high_qubit" / _dt.datetime.now().strftime("%Y%m%d-%H%M%S")))
    ap.add_argument("--storage-dir", type=str, default=str(script_dir / "state_tmp"))

    ap.add_argument(
        "--rerun",
        action="store_true",
        help="Rerun even if a successful matching entry already exists in the results JSON",
    )

    ap.add_argument("--jetson-clocks", action="store_true", default=True)
    ap.add_argument("--no-jetson-clocks", action="store_true")

    ap.add_argument("--cleanup-storage", action="store_true", default=True)
    ap.add_argument("--no-cleanup-storage", action="store_true")
    ap.add_argument("--keep-storage-on-failure", action="store_true", default=True)
    ap.add_argument("--delete-storage-on-failure", action="store_true")

    ap.add_argument(
        "--assumed-init-compression-ratio",
        type=float,
        default=0.004,
        help="Disk pre-check ratio (default 0.4%%). Used to skip likely-impossible qubits before starting.",
    )
    ap.add_argument("--force-disk", action="store_true", help="Ignore disk pre-check and try anyway")

    args = ap.parse_args()

    force_mode = bool(args.force_mode) and not bool(args.no_force_mode)
    jetson_clocks = bool(args.jetson_clocks) and not bool(args.no_jetson_clocks)
    cleanup_on_success = bool(args.cleanup_storage) and not bool(args.no_cleanup_storage)
    keep_storage_on_failure = bool(args.keep_storage_on_failure) and not bool(args.delete_storage_on_failure)

    binary = Path(args.binary)
    results_path = Path(args.results_file)
    log_dir = Path(args.log_dir)
    storage_dir = Path(args.storage_dir)
    storage_dir.mkdir(parents=True, exist_ok=True)

    if not binary.exists():
        log(f"ERROR: binary not found: {binary}")
        return 1

    if jetson_clocks:
        log("Applying jetson_clocks...")
        try_jetson_clocks()

    # Disk pre-check for each qubit (init-phase worst-case-ish). This avoids obvious failures.
    free0 = disk_free_bytes(storage_dir)
    log(f"Disk free under storage-dir: {free0/1024/1024/1024:.1f} GiB")

    for q in args.qubits:
        need = predicted_needed_bytes(q, args.assumed_init_compression_ratio)
        free = disk_free_bytes(storage_dir)
        if (not args.force_disk) and need > free:
            log(
                f"[SKIP-Q] {q}q predicted init disk need {need/1024/1024/1024:.1f} GiB "
                f"> free {free/1024/1024/1024:.1f} GiB (ratio={args.assumed_init_compression_ratio:.4f}). "
                f"Use --force-disk to attempt anyway."
            )
            continue

        storage_path = storage_dir / f"state_{q}q.bin"
        for circuit in args.circuits:
            run_one(
                binary=binary,
                qubits=q,
                circuit=circuit,
                depth=args.depth,
                storage_path=storage_path,
                results_path=results_path,
                log_dir=log_dir,
                sim_mode=args.sim_mode,
                force_mode=force_mode,
                cleanup_on_success=cleanup_on_success,
                keep_storage_on_failure=keep_storage_on_failure,
                rerun=bool(args.rerun),
            )

    log("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
