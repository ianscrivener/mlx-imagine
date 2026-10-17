"""Cache-limit measurement (ROADMAP T-10): one load and N generate jobs against a running server. Each job prints
its time, the MLX cache and active memory, and the swap in use. Run it only through cache_limit.sh, which starts a
fresh server with one cache limit, under the memory watchdog.

Usage: cache_limit.py MODEL JOBS [URL]"""

import json
import subprocess
import sys
import time
import urllib.request

MODEL, JOBS = sys.argv[1], int(sys.argv[2])
URL = sys.argv[3] if len(sys.argv) > 3 else "http://127.0.0.1:8765"
PROMPTS = ["a lighthouse in a storm", "a red fox in snow", "a bowl of ramen", "an old library", "a desert road at dusk"]


def call(method: str, path: str, body: dict | None = None) -> dict:
    data = json.dumps(body).encode() if body is not None else None
    request = urllib.request.Request(URL + path, data=data, method=method, headers={"content-type": "application/json"})
    with urllib.request.urlopen(request, timeout=900) as response:
        return json.loads(response.read())


def finish(job_id: str) -> dict:
    while True:
        job = call("GET", f"/v1/jobs/{job_id}")
        if job["state"] in ("succeeded", "failed", "cancelled"):
            return job
        time.sleep(0.3)


def swap_mb() -> int:
    used = subprocess.run(["sysctl", "-n", "vm.swapusage"], capture_output=True, text=True).stdout
    return int(float(used.split("used =")[1].split("M")[0]))


loaded = finish(call("POST", "/v1/model/load", {"model": MODEL})["id"])
if loaded["state"] != "succeeded":
    sys.exit(f"FAIL: the load job {loaded['state']}: {loaded.get('error')}")
print(f"load {loaded['state']} cache_limit={call('GET', '/v1/server')['memory']['mlx_cache_limit_gb']} GB", flush=True)
times = []
for index in range(JOBS):
    body = {"prompt": PROMPTS[index % len(PROMPTS)], "seed": [100 + index], "width": 512, "height": 512, "steps": 9}
    started = time.monotonic()
    job = finish(call("POST", "/v1/generate", body)["id"])
    seconds = time.monotonic() - started
    if job["state"] != "succeeded":
        sys.exit(f"FAIL: job {index + 1} {job['state']}: {job.get('error')}")
    memory = call("GET", "/v1/server")["memory"]
    times.append(seconds)
    print(
        f"job {index + 1:2d} {job['state']} {seconds:5.1f}s cache={memory['mlx_cache_gb']}GB "
        f"active={memory['mlx_active_gb']}GB peak={memory['mlx_peak_gb']}GB swap={swap_mb()}M",
        flush=True,
    )
print(
    f"first={times[0]:.1f}s last={times[-1]:.1f}s mean={sum(times) / len(times):.1f}s slowdown={times[-1] / times[0]:.2f}x"
)
