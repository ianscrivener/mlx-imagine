"""The mlx-imagine command: serve, mcp and service (docs/api.md section 7).

The config is read and its environment is set before any module that imports mflux loads (docs/scope.md 3.9)."""

import argparse
import json
import sys
import urllib.request
from pathlib import Path

from . import config


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="mlx-imagine", description="A long-running HTTP and MCP server for mflux.")
    parser.add_argument("--config", type=Path, help="The config file. Default: $MLX_IMAGINE_CONFIG or "
                        "~/.config/mlx-imagine/config.yaml.")  # fmt: skip
    commands = parser.add_subparsers(dest="command")

    serve = commands.add_parser("serve", help="Start the server (the default command).")
    serve.add_argument("--host", help="The address to bind to, for this run only.")
    serve.add_argument("--port", type=int, help="The port, for this run only.")

    commands.add_parser("mcp", help="Start the MCP server on stdio.")

    service = commands.add_parser("service", help="macOS only: manage the LaunchAgent.")
    service_commands = service.add_subparsers(dest="service_command", required=True)
    install = service_commands.add_parser("install", help="Install the LaunchAgent and start the server.")
    install.add_argument("--force", action="store_true", help="Install again over an installed LaunchAgent.")
    service_commands.add_parser("uninstall", help="Stop the server and delete the LaunchAgent.")
    service_commands.add_parser("status", help="Show the LaunchAgent state and the server health.")
    service_commands.add_parser("start", help="Start the server. Install the LaunchAgent first when needed.")
    service_commands.add_parser("stop", help="Stop the server. Keep the LaunchAgent installed.")
    service_commands.add_parser("restart", help="Restart the server.")
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    config_file = (args.config or config.config_path()).expanduser().resolve()
    try:
        settings = config.load(config_file)
    except Exception as exc:  # noqa: BLE001 -- a broken config file stops the command with its reason
        parser.error(f"cannot read the config file {config_file}: {exc}")
    command = args.command or "serve"

    if command == "service":
        _service(args, settings, config_file)
        return

    if command == "serve":
        if args.host:
            settings.host = args.host
        if args.port:
            settings.port = args.port
    if command in {"serve", "mcp"}:
        try:
            config.check_startup(settings)
        except Exception as exc:  # noqa: BLE001 -- ApiError: a config the server must not start with
            parser.error(str(exc))

    config.apply_environment(settings)
    config.make_directories(settings)
    import os

    os.environ["MLX_IMAGINE_CONFIG"] = str(config_file)

    if command == "mcp":
        _mcp_stdio(settings)
    else:
        _serve(settings)


def _serve(settings: config.Settings) -> None:
    import uvicorn

    from .api import create_app
    from .engine import Engine
    from .mcp_server import build_mcp

    _log_startup(settings)
    engine = Engine(settings)
    if settings.mcp_enable:
        mcp = build_mcp(engine)
        mcp_app = mcp.streamable_http_app(streamable_http_path="/mcp", host=settings.host)
        app = create_app(engine, mcp_app=mcp_app, mcp_lifespan=mcp.session_manager.run)
    else:
        app = create_app(engine)
    uvicorn.run(app, host=settings.host, port=settings.port, log_level="info")


def _log_startup(settings: config.Settings) -> None:
    import logging

    log = logging.getLogger("mlx-imagine")
    base = f"http://{settings.host}:{settings.port}"
    services = [
        f"{name} {base}{path}"
        for name, enabled, path in (
            ("REST API", settings.api_enable, "/v1"),
            ("MCP", settings.mcp_enable, "/mcp"),
            ("Swagger", settings.swagger_enable, "/docs"),
        )
        if enabled
    ]
    log.warning("Services: %s", "; ".join(services) or "none")
    if not settings.token:
        log.warning(
            "No API token is on (server.require_api_token is false). Each program on this Mac can use the server."
        )


def _mcp_stdio(settings: config.Settings) -> None:
    from .engine import Engine
    from .mcp_server import build_mcp

    url = f"http://{settings.host}:{settings.port}"
    if _healthy(f"{url}/v1/health"):
        # Two processes would each load a model. Point the client at the server that runs instead.
        sys.exit(f"mlx-imagine: a server runs at {url}. Connect the MCP client to {url}/mcp instead.")
    engine = Engine(settings)
    engine.start()
    try:
        build_mcp(engine).run()
    finally:
        engine.stop()


def _service(args: argparse.Namespace, settings: config.Settings, config_file: Path) -> None:
    from . import service

    try:
        if args.service_command == "install":
            path = service.install(config_file, force=args.force)
            print(f"Installed {path}. The server starts now and at each login.")
        elif args.service_command == "uninstall":
            path = service.uninstall()
            print(f"Stopped the server and deleted {path}.")
        elif args.service_command in ("start", "stop", "restart"):
            if args.service_command == "stop":
                result = service.stop()
            else:
                result = getattr(service, args.service_command)(config_file)
            print(f"The server is {result}.")
        else:
            print(json.dumps(service.status(f"http://{settings.host}:{settings.port}/v1/health"), indent=2))
    except service.ServiceError as exc:
        sys.exit(f"mlx-imagine: {exc}")


def _healthy(url: str) -> bool:
    try:
        with urllib.request.urlopen(url, timeout=1) as response:  # noqa: S310 -- a local http URL we build
            return response.status == 200
    except OSError:
        return False


if __name__ == "__main__":
    main()
