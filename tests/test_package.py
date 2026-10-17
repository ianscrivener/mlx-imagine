"""The package layout: one top-level package, mlx_imagine, next to mflux and not inside it."""

from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"


def test_the_source_tree_ships_only_mlx_imagine() -> None:
    """Bug: an old src/mflux/ folder stays in the tree, and an editable install puts it in front of the real mflux,
    so `import mflux` finds the wrong package."""
    assert sorted(path.name for path in SRC.iterdir() if not path.name.startswith(".")) == ["mlx_imagine"]


def test_the_package_imports_next_to_mflux() -> None:
    """Bug: the package imports a module of its own under the mflux name and shadows the library."""
    import mflux
    import mlx_imagine

    assert Path(mlx_imagine.__file__).parent == SRC / "mlx_imagine"
    assert SRC not in Path(mflux.__file__).parents
