import sys, os, re, torch
sys.path.append('.')
from PIL import Image
from app import all_aus, EMOTION_MAP, transform, device, model, canonical_au

test_dir = r"D:\Research4\TestDb\DATASET\test"

test_images = {
    '1': ['test_1138_aligned.jpg', 'test_1966_aligned.jpg', 'test_1204_aligned.jpg', 'test_2234_aligned.jpg', 'test_1750_aligned.jpg'],
    '2': ['test_1792_aligned.jpg', 'test_2280_aligned.jpg', 'test_2265_aligned.jpg', 'test_2277_aligned.jpg', 'test_2240_aligned.jpg'],
    '3': ['test_2339_aligned.jpg', 'test_2362_aligned.jpg', 'test_1627_aligned.jpg', 'test_2313_aligned.jpg', 'test_2383_aligned.jpg'],
    '4': ['test_2107_aligned.jpg', 'test_1965_aligned.jpg', 'test_1991_aligned.jpg', 'test_1353_aligned.jpg', 'test_2197_aligned.jpg'],
    '5': ['test_2158_aligned.jpg', 'test_1709_aligned.jpg', 'test_1688_aligned.jpg', 'test_0931_aligned.jpg', 'test_1537_aligned.jpg'],
    '6': ['test_1791_aligned.jpg', 'test_1301_aligned.jpg', 'test_2099_aligned.jpg', 'test_1669_aligned.jpg', 'test_1190_aligned.jpg'],
    '7': ['test_2452_aligned.jpg', 'test_2852_aligned.jpg', 'test_2616_aligned.jpg', 'test_2475_aligned.jpg', 'test_2986_aligned.jpg'],
}

data = []
for folder, imgs in test_images.items():
    gt = EMOTION_MAP[folder]
    for img_name in imgs:
        p = os.path.join(test_dir, folder, img_name)
        img = Image.open(p).convert('RGB')
        t = transform(img).unsqueeze(0).to(device)
        with torch.no_grad():
            probs = torch.sigmoid(model(t))[0].cpu().numpy()
        au_probs = {all_aus[i]: float(probs[i]) for i in range(len(all_aus))}
        data.append({'img': img_name, 'gt': gt, 'folder': folder, 'au_probs': au_probs})

import pickle
os.makedirs('scratch', exist_ok=True)
with open('scratch/data_35.pkl', 'wb') as f:
    pickle.dump(data, f)
print(f"Loaded and saved {len(data)} benchmark images!")
