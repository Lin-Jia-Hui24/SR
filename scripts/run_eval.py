#!/usr/bin/env python
"""Run standard and Self-Ensemble x8 evaluation for all ablation checkpoints.

For every config in configs/repro/<EXP>_seed<seed>.yaml (EXP in L0/P1/P2,
seed in 42/43/44) this runs evaluate_edsr.py twice per checkpoint (standard and
--self-ensemble), balancing jobs across GPUs by free memory.

Usage:
    .venv/bin/python scripts/run_eval.py
    .venv/bin/python scripts/run_eval.py --exps L0 P1 --seeds 42 43
"""
import argparse
import os
import subprocess
import threading
import queue
import time

MIN_FREE_MB = 10000
NUM_WORKERS = 4


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
gpu_jobs = {0: 0, 1: 0}


def pick_gpu():
    with lock:
        mem = free_mem()
        cands = [g for g in (0, 1) if mem.get(g, 0) >= MIN_FREE_MB]
        if not cands:
            return None
        g = min(cands, key=lambda x: (gpu_jobs[x], -mem[x]))
        gpu_jobs[g] += 1
        return g


def release_gpu(g):
    with lock:
        gpu_jobs[g] = max(0, gpu_jobs[g] - 1)


def make_jobs(exps, seeds):
    jobs = []
    for exp in exps:
        for seed in seeds:
            base = f"{exp}_seed{seed}"
            cfg = f"configs/repro/{base}.yaml"
            ckpt = f"outputs/repro/{base}/best.pt"
            jobs.append((base, cfg, ckpt, False))
            jobs.append((base + "_se", cfg, ckpt, True))
    return jobs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exps", nargs="+", default=["L0", "P1", "P2"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    args = ap.parse_args()

    q = queue.Queue()
    for j in make_jobs(args.exps, args.seeds):
        q.put(j)

    def worker(wid):
        while True:
            try:
                name, cfg, ckpt, se = q.get_nowait()
            except queue.Empty:
                return
            while True:
                gpu = pick_gpu()
                if gpu is not None:
                    break
                time.sleep(60)
            out = f"results/repro/{name}"
            cmd = [".venv/bin/python", "evaluate_edsr.py", "--config", cfg, "--checkpoint", ckpt, "--output", out]
            if se:
                cmd.append("--self-ensemble")
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu))
            logf = open(f"logs/repro/eval_{name}.log", "wb")
            proc = subprocess.Popen(cmd, stdout=logf, stderr=subprocess.STDOUT, env=env)
            print(f"[{now()}] START gpu={gpu} {name} se={se}", flush=True)
            rc = proc.wait()
            logf.close()
            release_gpu(gpu)
            print(f"[{now()}] DONE gpu={gpu} {name} rc={rc}", flush=True)
            if rc != 0:
                print(f"[{now()}] RETRY {name} rc={rc}", flush=True)
                time.sleep(60)
                q.put((name, cfg, ckpt, se))
            time.sleep(5)

    threads = [threading.Thread(target=worker, args=(w,), daemon=True) for w in range(NUM_WORKERS)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    print(f"[{now()}] ALL EVAL DONE", flush=True)


if __name__ == "__main__":
    main()
