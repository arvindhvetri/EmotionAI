import pickle, sys
sys.path.append('.')
with open('scratch/data_rep4.pkl', 'rb') as f:
    data = pickle.load(f)

from app import canonical_au

contempt_fakes = ['test_0023_aligned.jpg', 'test_2388_aligned.jpg', 'test_0774_aligned.jpg', 'test_0843_aligned.jpg', 'test_1899_aligned.jpg', 'test_0917_aligned.jpg', 'test_1116_aligned.jpg']

for item in data:
    if item['img'] in contempt_fakes:
        au_probs = item['au_probs']
        l12 = au_probs.get('L12', 0.0)
        r12 = au_probs.get('R12', 0.0)
        l14 = au_probs.get('L14', 0.0)
        r14 = au_probs.get('R14', 0.0)
        c_probs = {canonical_au(k): v for k, v in au_probs.items()}
        top = sorted([(canonical_au(k), round(v, 2)) for k, v in au_probs.items() if v > 0.3], key=lambda x: x[1], reverse=True)[:8]
        print(f"=== {item['img']} (GT: {item['gt']}) ===")
        print(f"  L12: {l12:.2f}, R12: {r12:.2f}, L14: {l14:.2f}, R14: {r14:.2f}")
        print(f"  Top AUs: {top}")
