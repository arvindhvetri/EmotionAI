import pickle, re, numpy as np
from sklearn.linear_model import LogisticRegression
from scratch_inspect import data as orig_35

with open('scratch/val_175.pkl', 'rb') as f:
    samples = pickle.load(f)

def canonical_au(raw_name):
    m = re.match(r'^(?:[LRBTt]{1,2}|AD|BL|RB)?(\d+)[A-Za-z]?$', str(raw_name))
    if m:
        return m.group(1)
    return str(raw_name)

canonical_keys = sorted(list({canonical_au(k) for s in samples for k in s['probs'].keys() if canonical_au(k).isdigit()}), key=lambda x: int(x))

def extract_features(au_probs):
    c_probs = {}
    for k, v in au_probs.items():
        c = canonical_au(k)
        c_probs[c] = max(c_probs.get(c, 0.0), v)
    feats = [c_probs.get(c, 0.0) for c in canonical_keys]
    l12 = au_probs.get('L12', 0.0)
    r12 = au_probs.get('R12', 0.0)
    l14 = au_probs.get('L14', 0.0)
    r14 = au_probs.get('R14', 0.0)
    feats.append(abs(l12 - r12))
    feats.append(abs(l14 - r14))
    return np.array(feats)

X = np.array([extract_features(s['probs']) for s in samples])
y = np.array([s['gt'] for s in samples])

clf = LogisticRegression(C=1.0, max_iter=1000)
clf.fit(X, y)

correct = 0
for item in orig_35:
    feat = extract_features(item['au_probs']).reshape(1, -1)
    pred = clf.predict(feat)[0]
    match = (pred == item['gt'])
    if match: correct += 1
    m_str = 'CORRECT' if match else 'MISMATCH'
    img_name = item['img']
    gt_name = item['gt']
    print(f"{img_name:25s} | GT: {gt_name:10s} | Pred: {pred:10s} | {m_str}")

print(f"\nAccuracy on original 35 report images: {correct/len(orig_35)*100:.1f}% ({correct}/{len(orig_35)})")
