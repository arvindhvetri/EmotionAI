import pickle, re
from collections import defaultdict

with open('scratch/val_175.pkl', 'rb') as f:
    samples = pickle.load(f)

def canonical_au(raw_name):
    m = re.match(r'^(?:[LRBTt]{1,2}|AD|BL|RB)?(\d+)[A-Za-z]?$', str(raw_name))
    if m:
        return m.group(1)
    return str(raw_name)

by_emo = defaultdict(lambda: defaultdict(list))
for s in samples:
    c_probs = {}
    for k, v in s['probs'].items():
        c = canonical_au(k)
        c_probs[c] = max(c_probs.get(c, 0.0), v)
    for c, v in c_probs.items():
        by_emo[s['gt']][c].append(v)

print("=== AVERAGE CANONICAL AU ACTIVATION PER EMOTION ===")
for emo, aus in sorted(by_emo.items()):
    avg = [(c, sum(vals)/len(vals)) for c, vals in aus.items()]
    top = sorted(avg, key=lambda x: x[1], reverse=True)[:8]
    top_str = ' '.join([f"AU{c}:{v:.2f}" for c, v in top])
    count = len(aus['1'])
    print(f"{emo:12s} ({count} imgs): {top_str}")
