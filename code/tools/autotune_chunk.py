#!/usr/bin/env python3
import os
import re
import subprocess
import sys

EDGE_BIN = "./build/edge_quantum"


def run_case(chunk_pow, qubits=28, circuit="VQE", depth=3, timeout_s=300):
    env = os.environ.copy()
    env["EDGEQ_CHUNK_POW"] = str(chunk_pow)
    env["EDGEQ_HANG_TIMEOUT"] = env.get("EDGEQ_HANG_TIMEOUT", "0")
    cmd = [EDGE_BIN, "--qubits", str(qubits), "--circuit", circuit, "--depth", str(depth), "--sim-mode", "async", "--force-mode"]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=timeout_s)
    except subprocess.TimeoutExpired:
        return 124, None, "TIMEOUT"
    out = proc.stdout + "\n" + proc.stderr
    m = re.search(r"Total Time:\s*([0-9.]+)", out)
    t = float(m.group(1)) if m else None
    return proc.returncode, t, out


def main():
    qubits = int(sys.argv[1]) if len(sys.argv) > 1 else 28
    circuit = sys.argv[2] if len(sys.argv) > 2 else "VQE"
    depth = int(sys.argv[3]) if len(sys.argv) > 3 else 3
    timeout_s = int(sys.argv[4]) if len(sys.argv) > 4 else 300
    powers = [25, 26]

    results = []
    for p in powers:
        code, t, out = run_case(p, qubits, circuit, depth, timeout_s=timeout_s)
        status = "OK" if code == 0 and t is not None else "FAIL"
        print(f"chunk_pow={p} => {status} time={t}")
        if status != "OK":
            tail = "\n".join(out.splitlines()[-5:]) if out else ""
            if tail:
                print("  last lines:")
                print("  " + "\n  ".join(tail.splitlines()))
        results.append((p, t, status))

    ok = [r for r in results if r[2] == "OK" and r[1] is not None]
    if ok:
        best = min(ok, key=lambda x: x[1])
        print(f"BEST: chunk_pow={best[0]} time={best[1]}")
    else:
        print("No successful runs.")


if __name__ == "__main__":
    main()
