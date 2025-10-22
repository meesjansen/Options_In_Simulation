# my_assets: packaged assets + helpers to resolve their paths
from importlib.resources import files, as_file
from contextlib import contextmanager
from typing import Iterator, Union
from pathlib import Path

# Package root as a Traversable
_ASSETS = files(__package__)

@contextmanager
def usd_path(*rel: Union[str, Path]) -> Iterator[Path]:
    """
    Yields a filesystem Path to a packaged asset (works installed or from source).
    Usage:
        with usd_path("urdf", "origin_v19_c", "origin_v19", "origin_v19.usd") as p:
            use(str(p))
    """
    resource = _ASSETS.joinpath(*map(str, rel))
    with as_file(resource) as p:
        yield p  # p is a pathlib.Path pointing to a real file

# Convenience for your specific file
@contextmanager
def origin_v18_usd() -> Iterator[Path]:
    # v18 uses the "origin_sym" USD
    with usd_path("urdf", "origin_sym", "origin_sym.usd") as p:
        yield p


@contextmanager
def origin_v19_usd() -> Iterator[Path]:
    with usd_path("urdf", "origin_v19_c", "origin_v19", "origin_v19.usd") as p:
        yield p