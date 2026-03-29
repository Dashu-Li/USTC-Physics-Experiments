import sys
import os

print(f"Python executable: {sys.executable}")
print(f"Python version: {sys.version}")
print("sys.path:")
for p in sys.path:
    print(f"  {p}")

print("-" * 20)
try:
    import clr
    print(f"Successfully imported clr: {clr}")
    try:
        from System import String
        print("Successfully imported System.String")
    except Exception as e:
        print(f"Failed to import from System: {e}")
except ImportError as e:
    print(f"Failed to import clr: {e}")
except Exception as e:
    print(f"An error occurred during import clr: {e}")

print("-" * 20)
try:
    import site
    print(f"User site packages: {site.getusersitepackages()}")
except Exception:
    pass
