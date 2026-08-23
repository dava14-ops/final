import sys
sys.path.insert(0, "C:/workspace")
from benchmarks.canonical_schema import CanonicalFleetDataset, DataProvenance

ds = CanonicalFleetDataset(source="synthetic", tractors=[], windows=[], events=[])
report = ds.provenance_report()
for k, v in report.items():
    print(f"  {k}: {v!r}")

# Also test with PROPRIETARY_CLAIMS
ds2 = CanonicalFleetDataset(
    source="claims",
    provenance=DataProvenance.PROPRIETARY_CLAIMS,
    citation="owner fleet 2024",
    tractors=[],
    windows=[],
    events=[],
    source_metadata={"note": "real claims data"},
)
print("\n--- PROPRIETARY_CLAIMS ---")
for k, v in ds2.provenance_report().items():
    print(f"  {k}: {v!r}")

print("\nOK")
