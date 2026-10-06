import pickle, sys
sys.path.append('.')
from app import classify_emotion_from_aus

with open('scratch/val_175.pkl', 'rb') as f:
    val_data = pickle.load(f)

emotions = ['Surprise', 'Fear', 'Disgust', 'Happiness', 'Sadness', 'Anger', 'Contempt', 'Neutral / Unknown']
matrix = {gt: {p: 0 for p in emotions} for gt in emotions[:-1]}

for item in val_data:
    pred = classify_emotion_from_aus(item['probs'])
    gt = item['gt']
    matrix[gt][pred] = matrix[gt].get(pred, 0) + 1

header = f"{'GT':12s} " + " ".join([f"{e[:4]:>6s}" for e in emotions])
print(header)
print("-" * len(header))
for gt in emotions[:-1]:
    row = " ".join([f"{matrix[gt][p]:6d}" for p in emotions])
    print(f"{gt:12s} {row}")
