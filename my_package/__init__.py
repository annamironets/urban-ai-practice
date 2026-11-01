"""
Python package boilerplate for IDU.

This package serves as a boilerplate for developing Python packages
for Institute of Design and Urban Studies (IDU). It includes basic metadata
such as versioning, author information, and license details.
"""

from importlib.metadata import PackageNotFoundError, version

from .my_module import MyClass, my_function

try:
    __version__ = version("my_package")
except PackageNotFoundError:
    __version__ = "0.0.0"

__author__ = "Vasilii Starikov"
__email__ = "vasilstar97@gmail.com"
__credits__ = []
__license__ = "BSD-3"
__all__ = ["MyClass", "my_function", "__version__", "__author__", "__email__", "__credits__", "__license__"]
