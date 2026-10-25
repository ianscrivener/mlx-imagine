#!/usr/bin/env python3
"""Make a saved MFlux model folder load as a Hugging Face repo.

Example:
    python scripts/hf_cache_helper.py ~/foobar        -> <your-user>/foobar
    python scripts/hf_cache_helper.py ~/foobar baz    -> baz/foobar
    python scripts/hf_cache_helper.py ~/foobar --remove

The script creates a repo folder in the Hugging Face cache. Each model file
in the repo folder is a symlink to the file in the original folder.
No model data is copied.
"""

import argparse
import getpass
import hashlib
import os
import re
import shutil
import sys
from pathlib import Path

MARKER = ".mflux_alias"
NAME_PATTERN = re.compile(r"^[A-Za-z0-9._-]+$")


def hf_hub_cache() -> Path:
    if os.environ.get("HF_HUB_CACHE"):
        return Path(os.environ["HF_HUB_CACHE"]).expanduser()
    if os.environ.get("HF_HOME"):
        return Path(os.environ["HF_HOME"]).expanduser() / "hub"
    return Path.home() / ".cache" / "huggingface" / "hub"


def fake_revision(model_dir: Path) -> str:
    # 40 hex characters, like a git commit hash. Hugging Face does not check it.
    return hashlib.sha256(str(model_dir).encode()).hexdigest()[:40]


def model_files(model_dir: Path) -> list[Path]:
    files = []
    for root, dirs, names in os.walk(model_dir):
        dirs[:] = sorted(d for d in dirs if not d.startswith("."))
        files.extend(Path(root) / n for n in sorted(names) if not n.startswith("."))
    return files


def create_alias(model_dir: Path, repo_cache: Path, revision: str) -> int:
    snapshot = repo_cache / "snapshots" / revision
    if snapshot.exists():
        shutil.rmtree(snapshot)
    snapshot.mkdir(parents=True)
    count = 0
    for src in model_files(model_dir):
        dst = snapshot / src.relative_to(model_dir)
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.symlink_to(src)
        count += 1
    (repo_cache / "refs").mkdir(exist_ok=True)
    (repo_cache / "refs" / "main").write_text(revision)
    (repo_cache / MARKER).write_text(str(model_dir) + "\n")
    return count


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("model_dir", help="Folder that holds the saved MFlux model.")
    parser.add_argument("username", nargs="?", default=None, help="HF user name. Default: your system user name.")
    parser.add_argument("--remove", action="store_true", help="Delete the alias from the cache.")
    args = parser.parse_args()

    model_dir = Path(args.model_dir).expanduser().resolve()
    username = args.username or getpass.getuser()
    name = model_dir.name

    if not model_dir.is_dir():
        print(f"ERROR: Not a folder: {model_dir}", file=sys.stderr)
        return 1
    for part in (username, name):
        if not NAME_PATTERN.match(part):
            print(
                f"ERROR: '{part}' has characters that HF does not allow. Use only letters, numbers, '.', '_' and '-'.",
                file=sys.stderr,
            )
            return 1

    repo_id = f"{username}/{name}"
    repo_cache = hf_hub_cache() / f"models--{username}--{name}"

    if repo_cache.exists() and not (repo_cache / MARKER).exists():
        print(f"ERROR: {repo_cache} exists and was not made by this script. Not touching it.", file=sys.stderr)
        return 1

    if args.remove:
        if repo_cache.exists():
            shutil.rmtree(repo_cache)
            print(f"Removed {repo_id}")
        else:
            print(f"Nothing to remove for {repo_id}")
        return 0

    revision = fake_revision(model_dir)
    count = create_alias(model_dir, repo_cache, revision)
    print(f"Repo:     {repo_id}")
    print(f"Revision: {revision}")
    print(f"Files:    {count} linked")
    print(f"Cache:    {repo_cache}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
