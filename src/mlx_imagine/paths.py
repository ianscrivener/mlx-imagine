"""Path checks. Every path a client names must resolve inside one of the configured directories."""

from pathlib import Path

from .errors import invalid_request, path_not_allowed


def resolve_inside(base: Path, candidate: str | Path) -> Path:
    """The resolved path, or path_not_allowed when it is outside base. Symlinks resolve first, so a link
    inside base that points outside it is refused too."""
    base = base.expanduser().resolve()
    path = Path(candidate).expanduser()
    if not path.is_absolute():
        path = base / path
    path = path.resolve()
    if path != base and base not in path.parents:
        raise path_not_allowed(f"The path {str(candidate)!r} is outside {str(base)!r}.", path=str(candidate))
    return path


def safe_name(name: object) -> str:
    """A single directory name: a string, no separator, no '..', not empty, not hidden."""
    if not isinstance(name, str):
        raise invalid_request("The name must be a string. Use one directory name.", name=repr(name))
    if not name or name in {".", ".."} or "/" in name or "\\" in name or name.startswith(".") or "\0" in name:
        raise invalid_request(f"{name!r} is not a valid name. Use one directory name.", name=name)
    return name
