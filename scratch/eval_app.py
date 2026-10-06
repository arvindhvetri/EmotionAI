import pickle, sys
sys.path.append('.')
from app import classify_emotion_from_aus

with open('scratch/data_35.pkl', 'rb') as f:
    data = pickle.load(f)

corr = 0
for item in data:
    pred = classify_emotion_from_aus(item['au_probs'])
    m = (pred == item['gt'])
    if m: corr += 1
    res = 'CORRECT' if m else 'MISMATCH'
    print(f"[{item['folder']}] {item['img']:25s} | GT: {item['gt']:10s} | Pred: {pred:10s} | {res}")

pct = corr / len(data) * 100
print(f"\n==========================================")
print(f"Accuracy in app.py: {corr}/{len(data)} ({pct:.1f}%)")
print(f"==========================================")
