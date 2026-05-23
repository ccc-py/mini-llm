import json, glob, os

files = glob.glob('metrics_*.json')
rows = []
for f in files:
    with open(f) as fh:
        d = json.load(fh)
        rows.append(d)

if not rows:
    print("No metrics files found. Run methods first.")
    exit(1)

print("=" * 60)
print(f"  {'Method':<20} {'Seen(before)':>12} {'Seen(after)':>12} {'Unseen(before)':>14} {'Unseen(after)':>12}")
print("=" * 60)
for r in rows:
    print(f"  {r['method']:<20} {r['seen_before']:>10.1f}% {r['seen_after']:>10.1f}% {r['unseen_before']:>11.1f}% {r['unseen_after']:>10.1f}%")
print("=" * 60)
