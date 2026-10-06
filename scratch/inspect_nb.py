import json

with open('FA_EFER_ResNet18_VGGFace2.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

print('Total cells:', len(nb['cells']))
for i, cell in enumerate(nb['cells']):
    src = ''.join(cell.get('source', []))[:120].replace('\n', ' ')
    print(f"Cell {i:2d} ({cell.get('cell_type'):9s}): {src}")
