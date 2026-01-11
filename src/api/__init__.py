"""Compatibility shim for api module.

This module provides backward compatibility by re-exporting from deployment.api.

DEPRECATED: Use 'from deployment.api import ...' instead.
This shim will be removed in 2 releases.
"""

import sys
import warnings
import importlib
from types import ModuleType
from importlib.abc import MetaPathFinder
from importlib.util import spec_from_loader

# Make this module act as a package by setting __path__
__path__ = []

# Custom import finder to handle submodule imports
class ApiSubmoduleFinder(MetaPathFinder):
    """Custom finder for api.* submodules."""
    
    def find_spec(self, name, path, target=None):
        if name.startswith('api.') and name != 'api':
            # Redirect to deployment.api.*
            submodule_name = name.replace('api.', 'deployment.api.', 1)
            try:
                spec = importlib.util.find_spec(submodule_name)
                if spec is not None:
                    loader = importlib.util.LazyLoader(spec.loader)
                    return spec_from_loader(name, loader)
            except (ImportError, ValueError):
                pass
        return None

# Install the finder if not already installed
_finder_installed = False
for finder in sys.meta_path:
    if isinstance(finder, ApiSubmoduleFinder):
        _finder_installed = True
        break

if not _finder_installed:
    sys.meta_path.insert(0, ApiSubmoduleFinder())

# Create submodule proxies for backward compatibility
_submodules = [
    'app',
    'config',
    'entities',
    'exception_handlers',
    'exceptions',
    'extractors',
    'inference',
    'middleware',
    'model_loader',
    'models',
    'response_converters',
    'startup',
    'routes',
    'tools',
    'cli',
]

def _create_submodule_proxy(name: str) -> ModuleType:
    """Create a proxy module for a submodule."""
    try:
        module = importlib.import_module(f'deployment.api.{name}')
        return module
    except ImportError:
        return ModuleType(f'api.{name}')

# Register submodule proxies
_current_module = sys.modules[__name__]
for submodule_name in _submodules:
    submodule = _create_submodule_proxy(submodule_name)
    sys.modules[f'api.{submodule_name}'] = submodule
    setattr(_current_module, submodule_name, submodule)

warnings.warn(
    "api is deprecated, use deployment.api instead. "
    "This shim will be removed in 2 releases.",
    DeprecationWarning,
    stacklevel=2
)

# Re-export everything from deployment.api
from deployment.api import *  # noqa: F403, F401
