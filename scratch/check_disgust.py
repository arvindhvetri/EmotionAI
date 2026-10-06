import pickle, re

with open('scratch/data_35.pkl', 'rb') as f:
    data = pickle.load(f)

def canonical_au(raw_name):
    m = re.match(r'^(?:[LRBTt]{1,2}|AD|BL|RB)?(\d+)[A-Za-z]?$', str(raw_name))
    if m:
        return m.group(1)
    return str(raw_name)

print("--- Disgust items AU12 and AU6 ---")
for item in data:
    if item['gt'] == 'Disgust':
        c_probs = {canonical_au(k): float(v) for k, v in item['au_probs'].items()}
        print(item['img'], "AU12:", round(c_probs.get('12', 0), 2), "AU6:", round(c_probs.get('6', 0), 2), "AU9:", round(c_probs.get('9', 0), 2), "AU10:", round(c_probs.get('10', 0), 2))
