"""Real-weight checks against a running mlx-imagine. Each check prints one PASS or FAIL line.

Usage: suite.py STEPS MODEL IMAGE [URL]
  STEPS  comma-separated: load,seeds,cancel,img2img,mcp,lowram,save,unload
  MODEL  a saved model name in mflux_models.local_directory, or a registry name
  IMAGE  a PNG for the img2img step

This loads real weights. Run it only through scripts/real/run.sh (just test-real), which starts the memory
watchdog first."""

import json
import sys
import time
import urllib.error
import urllib.request
import uuid
from typing import Any

URL = sys.argv[4] if len(sys.argv) > 4 else "http://127.0.0.1:8765"
STEPS = sys.argv[1].split(",")
MODEL = sys.argv[2]
IMAGE = sys.argv[3]
SAVE_MODEL = "z-image-turbo"
SAVE_NAME = "mlx-imagine-real-test-q8"
SMALL = {"width": 512, "height": 512, "steps": 9}


def call(method: str, path: str, body: Any = None, headers: dict | None = None, raw: bytes | None = None):
    data = raw if raw is not None else (json.dumps(body).encode() if body is not None else None)
    all_headers = {} if raw is not None else {"content-type": "application/json"}
    all_headers.update(headers or {})
    request = urllib.request.Request(URL + path, data=data, method=method, headers=all_headers)
    try:
        with urllib.request.urlopen(request, timeout=900) as response:
            return response.status, response.read(), dict(response.headers)
    except urllib.error.HTTPError as error:
        return error.code, error.read(), dict(error.headers)


def api(method: str, path: str, body: Any = None) -> Any:
    _, data, _ = call(method, path, body)
    return json.loads(data) if data else None


def wait(job_id: str, until: Any = None, timeout: float = 900) -> dict:
    started = time.time()
    while time.time() - started < timeout:
        job = api("GET", f"/v1/jobs/{job_id}")
        if until is not None and until(job):
            return job
        if job["state"] in ("succeeded", "failed", "cancelled"):
            return job
        time.sleep(0.5)
    raise SystemExit(f"timeout: job {job_id}")


def submit(path: str, body: Any) -> dict:
    job = api("POST", path, body)
    if "id" not in job:
        raise RuntimeError(json.dumps(job))
    return wait(job["id"])


def memory() -> dict:
    return api("GET", "/v1/server")["memory"]


def report(name: str, ok: bool, detail: str) -> None:
    print(f"{'PASS' if ok else 'FAIL'} | {name} | {detail}", flush=True)


def images(job: dict) -> list[dict]:
    return (job.get("result") or {}).get("images", [])


def step_load() -> None:
    started = time.time()
    job = submit("/v1/model/load", {"model": MODEL})
    used = memory()
    report(
        "load",
        job["state"] == "succeeded",
        f"{time.time() - started:.1f}s active={used['mlx_active_gb']}GB err={job['error']}",
    )


def step_seeds() -> None:
    started = time.time()
    job = submit("/v1/generate", {"prompt": "a red fox in fresh snow, photo", "seed": [1, 2], **SMALL})
    used = memory()
    names = [image["path"].split("/")[-1] for image in images(job)]
    detail = f"{time.time() - started:.1f}s {names} peak={used['mlx_peak_gb']} cache={used['mlx_cache_gb']}"
    report("generate 2 seeds", job["state"] == "succeeded" and len(names) == 2, f"{detail} err={job['error']}")


def step_cancel() -> None:
    job = api("POST", "/v1/generate", {"prompt": "a lighthouse at night, photo", "seed": [10, 11, 12], **SMALL})

    def second_image_started(current: dict) -> bool:
        progress = current.get("progress") or {}
        return progress.get("image_index", 0) >= 1 and progress.get("step", 0) >= 2

    wait(job["id"], until=second_image_started)
    api("DELETE", f"/v1/jobs/{job['id']}")
    job = wait(job["id"])
    ok = job["state"] == "cancelled" and len(images(job)) == 1
    report("cancel during image 2 of 3", ok, f"state={job['state']} images={len(images(job))}")


def step_img2img() -> None:
    boundary = "----" + uuid.uuid4().hex
    head = f'--{boundary}\r\nContent-Disposition: form-data; name="file"; filename="in.png"\r\nContent-Type: image/png\r\n\r\n'
    with open(IMAGE, "rb") as file:
        body = head.encode() + file.read() + f"\r\n--{boundary}--\r\n".encode()
    content_type = {"content-type": f"multipart/form-data; boundary={boundary}"}
    status, data, _ = call("POST", "/v1/uploads", raw=body, headers=content_type)
    upload = json.loads(data)
    request = {"prompt": "a watercolor painting", "image_path": upload["upload_id"], "image_strength": 0.3, "seed": [5]}
    job = submit("/v1/generate", {**request, "steps": 9})
    sizes = [(image["width"], image["height"]) for image in images(job)]
    report(
        "img2img from upload",
        job["state"] == "succeeded" and len(sizes) == 1,
        f"upload={status} {sizes} err={job['error']}",
    )


def step_mcp() -> None:
    accept = {"accept": "application/json, text/event-stream"}

    def rpc(message: dict, session: str | None = None) -> tuple[list[dict], dict]:
        headers = dict(accept)
        if session:
            headers["mcp-session-id"] = session
        _, data, response_headers = call("POST", "/mcp", message, headers=headers)
        text = data.decode()
        events = [json.loads(line[5:]) for line in text.splitlines() if line.startswith("data:")]
        if not events and text.strip():
            events = [json.loads(text)]
        return events, response_headers

    init = {"protocolVersion": "2025-06-18", "capabilities": {}, "clientInfo": {"name": "suite", "version": "1"}}
    _, headers = rpc({"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": init})
    session = headers.get("mcp-session-id") or headers.get("Mcp-Session-Id")
    rpc({"jsonrpc": "2.0", "method": "notifications/initialized"}, session)
    arguments = {"prompt": "a hot air balloon over green hills, photo", "seed": [7], **SMALL, "wait_seconds": 300}
    params = {"name": "generate_z_image_turbo", "arguments": arguments, "_meta": {"progressToken": "p1"}}
    started = time.time()
    events, _ = rpc({"jsonrpc": "2.0", "id": 2, "method": "tools/call", "params": params}, session)
    result = next((event for event in events if event.get("id") == 2), {})
    progress = [event for event in events if event.get("method") == "notifications/progress"]
    content = (result.get("result") or {}).get("content", [])
    kinds = [item["type"] for item in content]
    job = json.loads(content[0]["text"]) if content else {}
    ok = "image" in kinds and job.get("state") == "succeeded"
    report("MCP generate over HTTP", ok, f"{time.time() - started:.1f}s content={kinds} progress={len(progress)}")


def step_lowram() -> None:
    started = time.time()
    job = submit("/v1/model/load", {"model": MODEL, "low_ram": True})
    report("load low_ram", job["state"] == "succeeded", f"{time.time() - started:.1f}s err={job['error']}")
    for number in (1, 2):
        started = time.time()
        job = submit("/v1/generate", {"prompt": "a cup of coffee on a wooden table, photo", "seed": [number], **SMALL})
        model = api("GET", "/v1/model")
        used = memory()
        detail = f"{time.time() - started:.1f}s state={model and model['state']} peak={used['mlx_peak_gb']}"
        report(f"low_ram generate {number}", job["state"] == "succeeded", f"{detail} err={job['error']}")


def step_save() -> None:
    started = time.time()
    job = api("POST", "/v1/models/mflux/saves", {"model": SAVE_MODEL, "quantize": 8, "name": SAVE_NAME})
    if "id" not in job:
        report("save and quantize q8", False, json.dumps(job))
        return
    job = wait(job["id"], timeout=3600)
    used = memory()
    report(
        "save and quantize q8",
        job["state"] == "succeeded",
        f"{time.time() - started:.0f}s peak={used['mlx_peak_gb']} err={job['error']}",
    )
    deleted = api("DELETE", f"/v1/models/mflux/mflux_dir/{SAVE_NAME}?confirm=true")
    report("delete the saved test model", "freed_gb" in (deleted or {}), json.dumps(deleted))


def step_unload() -> None:
    job = submit("/v1/model/unload", None)
    used = memory()
    ok = job["state"] == "succeeded" and used["mlx_active_gb"] < 0.5
    report("unload", ok, f"active={used['mlx_active_gb']} cache={used['mlx_cache_gb']}")


STEP_FUNCTIONS = {
    "load": step_load,
    "seeds": step_seeds,
    "cancel": step_cancel,
    "img2img": step_img2img,
    "mcp": step_mcp,
    "lowram": step_lowram,
    "save": step_save,
    "unload": step_unload,
}

for name in STEPS:
    if name not in STEP_FUNCTIONS:
        raise SystemExit(f"unknown step {name!r}. Steps: {', '.join(STEP_FUNCTIONS)}")
    try:
        STEP_FUNCTIONS[name]()
    except Exception as exc:  # noqa: BLE001 -- one failed step must not hide the next
        report(name, False, f"{type(exc).__name__}: {exc}")
