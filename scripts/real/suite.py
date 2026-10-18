"""Real-weight checks against a running mlx-imagine. Each check prints one PASS or FAIL line.

Usage: suite.py STEPS MODEL IMAGE [URL]
  STEPS  comma-separated: load,seeds,cancel,img2img,mcp,lowram,save,unload,cycles,lora,inpaint,control
  LORAS  (step lora) comma-separated LoRA names in the LoRA directory; LORA_QUANTIZE and LORA_BASE_MODEL are load fields, LORA_SCALE and LORA_STEPS tune the run
  MODEL  a saved model name in mflux_models.local_directory, or a registry name
  IMAGE  a PNG for the img2img step

This loads real weights. Run it only through scripts/real/run.sh (just test-real), which starts the memory
watchdog first."""

import json
import os
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
        job = api("GET", f"/v8/jobs/{job_id}")
        if until is not None and until(job):
            return job
        if job["state"] in ("succeeded", "failed", "cancelled"):
            return job
        time.sleep(0.5)
    raise SystemExit(f"timeout: job {job_id}")


def submit(path: str, body: Any) -> dict:
    method = ("PUT" if body is not None else "DELETE") if path == "/v8/engine/model" else "POST"
    job = api(method, path, body)
    if "id" not in job:
        raise RuntimeError(json.dumps(job))
    return wait(job["id"])


def memory() -> dict:
    return api("GET", "/v8/server")["memory"]


def report(name: str, ok: bool, detail: str) -> None:
    print(f"{'PASS' if ok else 'FAIL'} | {name} | {detail}", flush=True)


def images(job: dict) -> list[dict]:
    return (job.get("result") or {}).get("images", [])


def step_load() -> None:
    started = time.time()
    job = submit("/v8/engine/model", {"model": MODEL})
    used = memory()
    report(
        "load",
        job["state"] == "succeeded",
        f"{time.time() - started:.1f}s active={used['mlx_active_gb']}GB err={job['error']}",
    )


def step_seeds() -> None:
    started = time.time()
    job = submit("/v8/generate", {"prompt": "a red fox in fresh snow, photo", "seed": [1, 2], **SMALL})
    used = memory()
    names = [image["path"].split("/")[-1] for image in images(job)]
    detail = f"{time.time() - started:.1f}s {names} peak={used['mlx_peak_gb']} cache={used['mlx_cache_gb']}"
    report("generate 2 seeds", job["state"] == "succeeded" and len(names) == 2, f"{detail} err={job['error']}")


def step_cancel() -> None:
    job = api("POST", "/v8/generate", {"prompt": "a lighthouse at night, photo", "seed": [10, 11, 12], **SMALL})

    def second_image_started(current: dict) -> bool:
        progress = current.get("progress") or {}
        return progress.get("image_index", 0) >= 1 and progress.get("step", 0) >= 2

    wait(job["id"], until=second_image_started)
    api("DELETE", f"/v8/jobs/{job['id']}")
    job = wait(job["id"])
    ok = job["state"] == "cancelled" and len(images(job)) == 1
    report("cancel during image 2 of 3", ok, f"state={job['state']} images={len(images(job))}")


def step_img2img() -> None:
    boundary = "----" + uuid.uuid4().hex
    head = f'--{boundary}\r\nContent-Disposition: form-data; name="file"; filename="in.png"\r\nContent-Type: image/png\r\n\r\n'
    with open(IMAGE, "rb") as file:
        body = head.encode() + file.read() + f"\r\n--{boundary}--\r\n".encode()
    content_type = {"content-type": f"multipart/form-data; boundary={boundary}"}
    status, data, _ = call("POST", "/v8/uploads", raw=body, headers=content_type)
    upload = json.loads(data)
    request = {"prompt": "a watercolor painting", "image_path": upload["upload_id"], "image_strength": 0.3, "seed": [5]}
    job = submit("/v8/generate", {**request, "steps": 9})
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
    job = submit("/v8/engine/model", {"model": MODEL, "low_ram": True})
    report("load low_ram", job["state"] == "succeeded", f"{time.time() - started:.1f}s err={job['error']}")
    for number in (1, 2):
        started = time.time()
        job = submit("/v8/generate", {"prompt": "a cup of coffee on a wooden table, photo", "seed": [number], **SMALL})
        model = api("GET", "/v8/engine/model")
        used = memory()
        detail = f"{time.time() - started:.1f}s state={model and model['state']} peak={used['mlx_peak_gb']}"
        report(f"low_ram generate {number}", job["state"] == "succeeded", f"{detail} err={job['error']}")


def step_save() -> None:
    started = time.time()
    job = api("POST", "/v8/saves", {"model": SAVE_MODEL, "quantize": 8, "name": SAVE_NAME})
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
    deleted = api("DELETE", f"/v8/models/mflux_dir/{SAVE_NAME}?confirm=true")
    report("delete the saved test model", "freed_gb" in (deleted or {}), json.dumps(deleted))


def step_unload() -> None:
    job = submit("/v8/engine/model", None)
    used = memory()
    ok = job["state"] == "succeeded" and used["mlx_active_gb"] < 0.5
    report("unload", ok, f"active={used['mlx_active_gb']} cache={used['mlx_cache_gb']}")


CYCLES = 8


def server_rss_gb() -> float | None:
    """The resident memory of the process that listens on the server port, or None when it cannot be read."""
    import subprocess

    port = URL.rsplit(":", 1)[-1]
    pid = subprocess.run(["lsof", "-ti", f"tcp:{port}", "-sTCP:LISTEN"], capture_output=True, text=True).stdout.split()
    if not pid:
        return None
    rss = subprocess.run(["ps", "-o", "rss=", "-p", pid[0]], capture_output=True, text=True).stdout.strip()
    return round(int(rss) / 1024**2, 2) if rss.isdigit() else None


def step_cycles() -> None:
    """Load, generate one small image and unload, many times. A leak shows as MLX memory or process memory that
    grows from cycle to cycle (docs/scope.md 3.8)."""
    active, rss = [], []
    for cycle in range(CYCLES):
        jobs = [
            submit("/v8/engine/model", {"model": MODEL}),
            submit("/v8/generate", {"prompt": "a red fox in fresh snow, photo", "seed": [cycle], **SMALL}),
            submit("/v8/engine/model", None),
        ]
        if any(job["state"] != "succeeded" for job in jobs):
            report("load/unload cycles", False, f"cycle {cycle + 1}: {[job['state'] for job in jobs]}")
            return
        used = memory()
        active.append(used["mlx_active_gb"])
        rss.append(server_rss_gb())
        print(
            f"  cycle {cycle + 1}: active={used['mlx_active_gb']}GB cache={used['mlx_cache_gb']}GB rss={rss[-1]}GB",
            flush=True,
        )
    known = [value for value in rss if value is not None]
    grew = bool(known) and known[-1] - known[1 if len(known) > 1 else 0] > 1.0
    measured = len(known) == len(rss)
    ok = max(active) < 0.5 and not grew and measured
    report(
        "load/unload cycles",
        ok,
        f"{CYCLES} cycles, active max={max(active)}GB, rss first={known[:1]} last={known[-1:]}"
        + ("" if measured else " (a process memory sample is missing: inconclusive)"),
    )


LORA_REQUEST = {"prompt": "close-up portrait photo of a woman, natural light, film grain", "seed": [7], "steps": 12}


def _lora_image(loras: list[str] | None) -> tuple[dict, str | None]:
    """Load MODEL with the LoRAs (or none), make one image with a fixed seed. Returns the job and the image path."""
    request: dict[str, Any] = {"model": MODEL}
    if os.environ.get("LORA_QUANTIZE"):  # not for a saved model, which is quantized already
        request["quantize"] = int(os.environ["LORA_QUANTIZE"])
    if os.environ.get("LORA_BASE_MODEL"):
        request["base_model"] = os.environ["LORA_BASE_MODEL"]
    if loras:
        scale = float(os.environ.get("LORA_SCALE", "1.0"))
        request["loras"] = [{"name": name, "scale": scale} for name in loras]
    job = submit("/v8/engine/model", request)
    if job["state"] != "succeeded":
        return job, None
    steps = int(os.environ.get("LORA_STEPS", LORA_REQUEST["steps"]))
    job = submit("/v8/generate", {**LORA_REQUEST, "steps": steps, "width": 768, "height": 768})
    found = images(job)
    return job, found[0]["path"] if found else None


def step_lora() -> None:
    """Each LoRA must change the image: the same seed with the LoRA must differ from the image without it."""
    from PIL import Image, ImageChops, ImageStat

    names = [name for name in os.environ.get("LORAS", "").split(",") if name]
    if not names:
        report("lora", False, "Set LORAS to a comma-separated list of LoRA names.")
        return
    listed = {item["name"] for item in api("GET", "/v8/loras")["loras"]}
    missing = [name for name in names if name not in listed]
    report("lora list", not missing, f"missing={missing}" if missing else f"{len(names)} LoRAs listed")
    if missing:
        return
    job, base = _lora_image(None)
    report("lora baseline", base is not None, f"{base} err={job['error']}")
    if base is None:
        return
    for name in names:
        started = time.time()
        job, path = _lora_image([name])
        if path is None:
            report(f"lora {name}", False, f"state={job['state']} err={job['error']}")
            continue
        with Image.open(base).convert("RGB") as plain, Image.open(path).convert("RGB") as styled:
            difference = sum(ImageStat.Stat(ImageChops.difference(plain, styled)).mean) / 3
        report(
            f"lora {name}",
            difference > 1.0,
            f"{time.time() - started:.0f}s mean pixel difference {difference:.1f} of 255, image {path}",
        )


def _upload_bytes(data: bytes) -> str:
    boundary = "----" + uuid.uuid4().hex
    head = f'--{boundary}\r\nContent-Disposition: form-data; name="file"; filename="in.png"\r\nContent-Type: image/png\r\n\r\n'
    body = head.encode() + data + f"\r\n--{boundary}--\r\n".encode()
    content_type = {"content-type": f"multipart/form-data; boundary={boundary}"}
    return json.loads(call("POST", "/v8/uploads", raw=body, headers=content_type)[1])["upload_id"]


def step_inpaint() -> None:
    """Qwen-Image 2.1 ControlNet: inpaint (a source image and a mask), then a control image. MODEL is a saved
    Qwen-Image 2.1 model or a registry name. INPAINT_BASE_MODEL names the ControlNet entry (the default)."""
    import io

    from PIL import Image, ImageChops, ImageStat

    base = os.environ.get("INPAINT_BASE_MODEL", "qwen-image-2.1-controlnet-union")
    with open(IMAGE, "rb") as file:
        source_bytes = file.read()
    with Image.open(io.BytesIO(source_bytes)) as source:
        width, height = source.size
    mask = Image.new("L", (width, height), 0)
    mask.paste(255, (width // 4, height // 4, width * 3 // 4, height * 3 // 4))  # the middle is drawn again
    buffer = io.BytesIO()
    mask.save(buffer, "PNG")
    image_id, mask_id = _upload_bytes(source_bytes), _upload_bytes(buffer.getvalue())

    status, data, headers = call("POST", "/v8/masks/preview", {"image_path": image_id, "mask_image": mask_id})
    report("mask preview", status == 200 and headers.get("content-type") == "image/png", f"{status} {len(data)} bytes")

    started = time.time()
    job = submit("/v8/engine/model", {"model": MODEL, "base_model": base})
    report("load with ControlNet", job["state"] == "succeeded", f"{time.time() - started:.0f}s err={job['error']}")
    if job["state"] != "succeeded":
        return
    request = {"prompt": "a bright red balloon floating", "seed": [3], "steps": 20}
    started = time.time()
    job = submit("/v8/generate", {**request, "image_path": image_id, "mask_image": mask_id})
    found = images(job)
    if not found:
        report("inpaint", False, f"state={job['state']} err={job['error']}")
    else:
        with Image.open(found[0]["path"]).convert("RGB") as result, Image.open(IMAGE).convert("RGB") as original:
            result = result.resize(original.size)
            difference = ImageChops.difference(result, original).convert("L")
            inside = ImageStat.Stat(difference, mask=mask).mean[0]
            outside = ImageStat.Stat(difference, mask=ImageChops.invert(mask)).mean[0]
        report(
            "inpaint",
            inside > 2 * outside and inside > 5,
            f"{time.time() - started:.0f}s changed inside={inside:.1f} outside={outside:.1f} image {found[0]['path']}",
        )
    started = time.time()
    job = submit("/v8/generate", {**request, "controlnet_image_path": image_id, "steps": 20})
    found = images(job)
    report(
        "control image",
        bool(found),
        f"{time.time() - started:.0f}s image {found[0]['path'] if found else None} err={job['error']}",
    )


def step_control() -> None:
    """Z-Image Turbo ControlNet Union: make an edge image of IMAGE, then generate with it as a canny control.
    MODEL is a saved Z-Image ControlNet model or a registry name. CONTROL_BASE_MODEL names the entry."""
    import io

    from PIL import Image, ImageFilter

    base = os.environ.get("CONTROL_BASE_MODEL", "z-image-controlnet")
    with Image.open(IMAGE) as source:
        edges = source.convert("L").resize((512, 512)).filter(ImageFilter.FIND_EDGES)
    buffer = io.BytesIO()
    edges.save(buffer, "PNG")
    control_id = _upload_bytes(buffer.getvalue())
    started = time.time()
    job = submit("/v8/engine/model", {"model": MODEL, "base_model": base})
    report("load with ControlNet", job["state"] == "succeeded", f"{time.time() - started:.0f}s err={job['error']}")
    if job["state"] != "succeeded":
        return
    started = time.time()
    request = {"prompt": "a puffin on a rock, photo", "seed": [4], "steps": 9, "width": 512, "height": 512}
    job = submit("/v8/generate", {**request, "control": [f"canny:{control_id}:0.8"]})
    found = images(job)
    report(
        "generate with a canny control",
        bool(found),
        f"{time.time() - started:.0f}s image {found[0]['path'] if found else None} err={job['error']}",
    )


STEP_FUNCTIONS = {
    "load": step_load,
    "seeds": step_seeds,
    "cancel": step_cancel,
    "img2img": step_img2img,
    "mcp": step_mcp,
    "lowram": step_lowram,
    "save": step_save,
    "unload": step_unload,
    "cycles": step_cycles,
    "lora": step_lora,
    "inpaint": step_inpaint,
    "control": step_control,
}

for name in STEPS:
    if name not in STEP_FUNCTIONS:
        raise SystemExit(f"unknown step {name!r}. Steps: {', '.join(STEP_FUNCTIONS)}")
    try:
        STEP_FUNCTIONS[name]()
    except Exception as exc:  # noqa: BLE001 -- one failed step must not hide the next
        report(name, False, f"{type(exc).__name__}: {exc}")
