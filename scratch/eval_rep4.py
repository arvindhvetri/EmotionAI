import pickle, sys
sys.path.append('.')
from app import classify_emotion_from_aus

with open('scratch/data_rep4.pkl', 'rb') as f:
    data = pickle.load(f)

corr = 0
by_folder = {}
for item in data:
    pred = classify_emotion_from_aus(item['au_probs'])
    m = (pred == item['gt'])
    if m: corr += 1
    res = 'CORRECT' if m else 'MISMATCH'
    fld = item['folder']
    if fld not in by_folder: by_folder[fld] = [0, 0, item['gt']]
    by_folder[fld][1] += 1
    if m: by_folder[fld][0] += 1
    print(f"[{item['folder']}] {item['img']:25s} | GT: {item['gt']:10s} | Pred: {pred:18s} | {res}")

print(f"\n==========================================")
print(f"Total Accuracy on Report 4: {corr}/{len(data)} ({corr/len(data)*100:.1f}%)")
print(f"==========================================")
for fld in sorted(by_folder.keys()):
    c, t, name = by_folder[fld]
    print(f"  Folder {fld} ({name:10s}): {c}/{t} ({c/t*100:.1f}%)")
