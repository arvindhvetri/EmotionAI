import os, pypdf

reader = pypdf.PdfReader('TestDb_Batch_Report (4).pdf')

items = []
current_gt = None
for i, page in enumerate(reader.pages[1:8]):
    text = page.extract_text()
    lines = text.split('\n')
    for idx, line in enumerate(lines):
        if 'Ground Truth:' in line and 'Folder' in line:
            current_gt = line.split('(')[0].replace('Ground Truth:', '').strip()
        if 'Image:' in line:
            img_file = line.replace('Image:', '').strip()
            pred = None
            res = None
            for j in range(idx, min(idx + 10, len(lines))):
                if 'Predicted Emotion:' in lines[j]:
                    pred = lines[j].replace('Predicted Emotion:', '').strip()
                if 'Result:' in lines[j]:
                    res = lines[j].replace('Result:', '').strip()
            items.append({'img': img_file, 'gt': current_gt, 'pred': pred, 'res': res})

print(f"Extracted {len(items)} items from Report (4):")
for it in items:
    print(f"[{it['gt']:10s}] {it['img']:25s} -> Pred: {str(it['pred']):15s} | {it['res']}")
