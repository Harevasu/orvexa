import glob
import hashlib
from pathlib import Path

files = sorted(
    glob.glob("artifacts/models/phase5/*.*")
    + glob.glob("artifacts/preprocessors/phase5/*.*")
    + glob.glob("data/processed/predictions/phase5/*.*")
    + glob.glob("reports/phase5/*.*")
)

print(f"{'SHA-256':<64}  {'Artifact Path'}")
print("-" * 100)
for p in files:
    h = hashlib.sha256(open(p, "rb").read()).hexdigest()
    print(f"{h}  {Path(p).as_posix()}")
