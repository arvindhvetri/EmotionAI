import os, pickle, torch, pypdf
from PIL import Image
import sys; sys.path.append('.')
from app import model, all_aus, transform, EMOTION_MAP, device

reader = pypdf.PdfReader('TestDb_Batch_Report (4).pdf')

items = []
current_gt = None
current_folder = None
test_dir = r"D:\Research4\TestDb\DATASET\test"

for i, page in enumerate(reader.pages[1:8]):
    folder_idx = str(i + 1)
    gt = EMOTION_MAP[folder_idx]
    text = page.extract_text()
    for line in text.split('\n'):
        if 'Image:' in line:
            img_name = line.replace('Image:', '').strip()
            img_path = os.path.join(test_dir, folder_idx, img_name)
            items.append({
                'img': img_name,
                'path': img_path,
                'folder': folder_idx,
                'gt': gt
            })

print(f"Loading and predicting {len(items)} images...")
data = []
with torch.no_grad():
    for item in items:
        img = Image.open(item['path']).convert('RGB')
        tensor = transform(img).unsqueeze(0).to(device)
        logits = model(tensor)
        probs = torch.sigmoid(logits).squeeze(0).cpu().numpy()
        au_probs = {all_aus[j]: float(probs[j]) for j in range(len(all_aus))}
        data.append({
            'img': item['img'],
            'folder': item['folder'],
            'gt': item['gt'],
            'au_probs': au_probs
        })

with open('scratch/data_rep4.pkl', 'wb') as f:
    pickle.dump(data, f)

print(f"Successfully cached {len(data)} images to scratch/data_rep4.pkl!")
