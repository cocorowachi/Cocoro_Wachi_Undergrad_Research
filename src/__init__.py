import os
import glob
import importlib

# Get a list of all Python files in the current folder
py_files = glob.glob(os.path.join(os.path.dirname(__file__), "*.py"))

py_files.sort(key=lambda x: (x != os.path.join(os.path.dirname(__file__), "imports.py"), x))

# Import all Python files dynamically, except __init__.py
for py_file in py_files:
    if py_file != os.path.join(os.path.dirname(__file__), "__init__.py"):
        module_name = os.path.basename(py_file)[:-3]  # Remove the .py extension
        module = importlib.import_module(f".{module_name}", package=__name__)

        # Dynamically import all functions from the module
        globals().update({name: getattr(module, name) for name in dir(module) if callable(getattr(module, name))})
