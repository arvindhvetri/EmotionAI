import pickle, re

with open('scratch/data_35.pkl', 'rb') as f:
    data = pickle.load(f)

def canonical_au(raw_name):
    m = re.match(r'^(?:[LRBTt]{1,2}|AD|BL|RB)?(\d+)[A-Za-z]?$', str(raw_name))
    if m:
        return m.group(1)
    return str(raw_name)

target_imgs = ['test_1792_aligned.jpg', 'test_2240_aligned.jpg', 'test_2313_aligned.jpg', 'test_1965_aligned.jpg', 'test_1537_aligned.jpg', 'test_2452_aligned.jpg']

for item in data:
    if item['img'] in target_imgs:
        c_probs = {}
        for au, pr in item['au_probs'].items():
            c = canonical_au(au)
            c_probs[c] = max(c_probs.get(c, 0.0), float(pr))
        top = sorted(c_probs.items(), key=lambda x: x[1], reverse=True)[:10]
        raw_top = sorted(item['au_probs'].items(), key=lambda x: x[1], reverse=True)[:8]
        print(f"=== {item['img']} (GT: {item['gt']}) ===")
        print("  Canonical Top:", [(k, round(v, 2)) for k, v in top])
        print("  Raw Top:", [(k, round(v, 2)) for k, v in raw_top])
        print("  Key AUs: AU1:", round(c_probs.get('1', 0), 2), "AU4:", round(c_probs.get('4', 0), 2), "AU6:", round(c_probs.get('6', 0), 2), "AU7:", round(c_probs.get('7', 0), 2), "AU9:", round(c_probs.get('9', 0), 2), "AU10:", round(c_probs.get('10', 0), 2), "AU12:", round(c_probs.get('12', 0), 2), "AU14:", round(c_probs.get('14', 0), 2), "AU16:", round(c_probs.get('16', 0), 2), "AU20:", round(c_probs.get('20', 0), 2), "AU23:", round(c_probs.get('23', 0), 2), "AU24:", round(c_probs.get('24', 0), 2), "AU25:", round(c_probs.get('25', 0), 2), "AU27:", round(c_probs.get('27', 0), 2))
