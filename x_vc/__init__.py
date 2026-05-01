from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("x-vc")
except PackageNotFoundError:
    __version__ = "unknown"
