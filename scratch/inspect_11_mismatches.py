import sys, os
sys.path.append('.')
import torch, re
from PIL import Image
from app import all_aus, EMOTION_MAP, transform, device, model, canonical_au

test_dir = r"D:\Research4\TestDb\DATASET\test"

mismatches = [
    ('2', 'test_1792_aligned.jpg', 'Fear'),
    ('2', 'test_2280_aligned.jpg', 'Fear'),
    ('2', 'test_2240_aligned.jpg', 'Fear'),
    ('3', 'test_2362_aligned.jpg', 'Disgust'),
    ('3', 'test_1627_aligned.jpg', 'Disgust'),
    ('3', 'test_2313_aligned.jpg', 'Disgust'),
    ('4', 'test_1353_aligned.jpg', 'Happiness'),
    ('5', 'test_1537_aligned.jpg', 'Sadness'),
    ('6', 'test_1669_aligned.jpg', 'Anger'),
    ('7', 'test_2452_aligned.jpg', 'Contempt'),
    ('7', 'test_2475_aligned.jpg', 'Contempt'),
]

print("=== DEEP DIVE ON THE 11 MISMATCHES ===")
for folder, img_name, gt in mismatches:
    p = os.path.join(test_dir, folder, img_name)
    img = Image.open(p).convert('RGB')
    t = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        probs = torch.sigmoid(model(t))[0].cpu().numpy()
    
    au_probs = {all_aus[i]: float(probs[i]) for i in range(len(all_aus))}
    c_probs = {}
    for k, v in au_probs.items():
        c = canonical_au(k)
        c_probs[c] = max(c_probs.get(c, 0.0), v)
    
    top_aus = sorted(c_probs.items(), key=lambda x: x[1], reverse=True)[:10]
    top_str = ' '.join([f"AU{k}:{v:.2f}" for k, v in top_aus if v > 0.3])
    
    raw_top = sorted(au_probs.items(), key=lambda x: x[1], reverse=True)[:8]
    raw_str = ' '.join([f"{k}:{v:.2f}" for k, v in raw_top if v > 0.3])
    
    print(f"\n[{folder}] {gt:10s} | {img_name}")
    print(f"  Canonical AUs (>0.3): {top_str}")
    print(f"  Raw AUs (>0.3)      : {raw_str}")
