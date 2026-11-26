"""Turn locust/results/c{1,2,4}_stats.csv into a markdown table and a chart."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

RESULTS = Path(__file__).resolve().parent.parent / "locust" / "results"
rows = []
for n in [1, 2, 4]:
    f = RESULTS / f"c{n}_stats.csv"
    if not f.exists():
        continue
    df = pd.read_csv(f)
    p = df[df["Name"] == "/predict"].iloc[0]
    rows.append({
        "containers": n,
        "requests": int(p["Request Count"]),
        "failures": int(p["Failure Count"]),
        "req/s": round(p["Requests/s"], 1),
        "avg ms": round(p["Average Response Time"]),
        "median ms": round(p["Median Response Time"]),
        "p95 ms": round(p["95%"]),
        "p99 ms": round(p["99%"]),
        "max ms": round(p["Max Response Time"]),
    })
table = pd.DataFrame(rows)
md = table.to_markdown(index=False)
(RESULTS / "summary.md").write_text(md + "\n")
print(md)

fig, axes = plt.subplots(1, 2, figsize=(11, 4))
axes[0].plot(table["containers"], table["median ms"], "o-", label="median")
axes[0].plot(table["containers"], table["p95 ms"], "o-", label="95th percentile")
axes[0].set(xlabel="API containers", ylabel="response time (ms)", title="/predict response time", xticks=[1, 2, 4])
axes[0].legend()
axes[1].bar(table["containers"].astype(str), table["req/s"], color="#2e6b34")
axes[1].set(xlabel="API containers", ylabel="requests per second", title="/predict throughput")
plt.tight_layout()
plt.savefig(RESULTS / "flood_test.png", dpi=120)
print("saved", RESULTS / "flood_test.png")
