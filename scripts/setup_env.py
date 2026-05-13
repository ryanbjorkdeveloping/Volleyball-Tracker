import os
import sys
from dotenv import load_dotenv

load_dotenv()

print("=== Environment Check ===\n")

# Check required env vars
required = ["ROBOFLOW_API_KEY", "ROBOFLOW_WORKSPACE", "ROBOFLOW_PROJECT", "ROBOFLOW_VERSION"]
all_ok = True
for key in required:
    val = os.environ.get(key)
    status = "OK" if val else "MISSING"
    display = f"{val[:4]}..." if val else "not set"
    print(f"  {key}: {status} ({display})")
    if not val:
        all_ok = False

print()

# Check Python packages
packages = ["ultralytics", "roboflow", "cv2", "numpy"]
for pkg in packages:
    try:
        mod = __import__(pkg)
        version = getattr(mod, "__version__", "installed")
        print(f"  {pkg}: {version}")
    except ImportError:
        print(f"  {pkg}: NOT INSTALLED")
        all_ok = False

print()

# Check GPU
try:
    import torch
    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none (CPU only)"
    print(f"  GPU: {gpu}")
except ImportError:
    print("  GPU: torch not installed, cannot check")

print()
if all_ok:
    print("Ready. Run: python scripts/download_dataset.py")
else:
    print("Fix the issues above before continuing.")
    sys.exit(1)
