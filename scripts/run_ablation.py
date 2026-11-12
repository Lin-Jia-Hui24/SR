#!/usr/bin/env python
"""Memory-aware training scheduler for EDSR ablation runs.

Runs the given config files across available GPUs, picking at launch time the
GPU with the most free memory (>= MIN_FREE_MB). A failed run is retried after
BACKOFF_S seconds instead of being dropped.

Usage:
    .venv/bin/python scripts/run_ablation.py configs/repro/L0_seed42.yaml configs/repro/L0_seed43.yaml ...
"""
import argparse
import os
import subprocess
import threading
import queue
import time

MIN_FREE_MB = 24000
BACKOFF_S = 600
NUM_WORKERS = 2


def now():
    return time.strftime("%Y-%m-%dT%H:%M:%S")


def free_mem():
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,memory.free", "--format=csv,noheader,nounits"],
        capture_output=True, text=True,
    ).stdout.strip().splitlines()
    mem = {}
    for line in out:
        if not line.strip():
            continue
        i, m = line.split(",")
        mem[int(i.strip())] = int(m.strip())
    return mem


lock = threading.Lock()
busy = set()


def reserve_gpu():
    while True:
        with lock:
            mem = free_mem()
            cands = [g for g in mem if g not in busy and mem[g] >= MIN_FREE_MB]
            if cands:
                g = max(cands, key=lambda x: mem[x])
                busy.add(g)
                return g, mem[g]
        time.sleep(60)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("configs", nargs="+", help="YAML config paths to train, in order.")
    args = ap.parse_args()

    q = queue.Queue()
    for cfg in args.configs:
        q.put(cfg)

    def worker(wid):
        while True:
            try:
                cfg = q.get_nowait()
            except queue.Empty:
                return
            gpu, free_mb = reserve_gpu()
            name = cfg.rsplit("/", 1)[-1][:-5]
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu))
            logf = open(f"logs/repro/{name}.log", "wb")
            proc = subprocess.Popen(
                [".venv/bin/python", "train.py", "--config", cfg, "--no-wandb"],
                stdout=logf, stderr=subprocess.STDOUT, env=env,
            )
            with open(f"logs/repro/{name}.pid", "w") as f:
                f.write(str(proc.pid) + "\n")
            print(f"[{now()}] START gpu={gpu} {name} pid={proc.pid} free={free_mb}MB", flush=True)
            rc = proc.wait()
            logf.close()
            with lock:
                busy.discard(gpu)
            print(f"[{now()}] DONE gpu={gpu} {name} rc={rc}", flush=True)
            if rc != 0:
                print(f"[{now()}] RETRY {name} rc={rc}", flush=True)
                time.sleep(BACKOFF_S)
                q.put(cfg)
            time.sleep(15)

    threads = [threading.Thread(target=worker, args=(w,), daemon=True) for w in range(NUM_WORKERS)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    print(f"[{now()}] ALL DONE", flush=True)


if __name__ == "__main__":
    main()
