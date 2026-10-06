import pickle, re

with open('scratch/data_35.pkl', 'rb') as f:
    data = pickle.load(f)

def canonical_au(raw_name):
    m = re.match(r'^(?:[LRBTt]{1,2}|AD|BL|RB)?(\d+)[A-Za-z]?$', str(raw_name))
    if m:
        return m.group(1)
    return str(raw_name)

def debug_v8(au_probs):
    c_probs = {}
    for au, pr in au_probs.items():
        c = canonical_au(au)
        c_probs[c] = max(c_probs.get(c, 0.0), float(pr))
    def p(c): return c_probs.get(str(c), 0.0)

    scores = {e: 0.0 for e in ['Surprise', 'Fear', 'Disgust', 'Happiness', 'Sadness', 'Anger', 'Contempt']}

    # Trace each rule
    # FEAR
    if p(27) > 0.7 and p(16) > 0.6:
        if (p(9) > 0.7 or p(10) > 0.85) and p(7) > 0.5 and p(1) < 0.4:
            scores['Anger'] += (p(27) + p(16) + max(p(9), p(10)) + p(7)) * 4.5
            print("  Fired Fear->Anger roar rule")
        elif p(9) < 0.6 or (p(20) > 0.5 and p(7) < 0.7) or p(4) < 0.5:
            scores['Fear'] += (p(27) + p(16)) * 6.5
            print("  Fired Fear scream rule")
            if p(1) > 0.4 or p(2) > 0.4: scores['Fear'] += 2.5
            if p(20) > 0.5: scores['Fear'] += p(20) * 3.0
        else:
            scores['Anger'] += (p(27) + p(16) + p(9) + p(7)) * 4.0

    # ANGER
    if p(16) > 0.7 and (p(25) > 0.7 or p(27) > 0.7) and p(9) > 0.7 and p(7) > 0.6:
        scores['Anger'] += (p(16) + max(p(25), p(27)) + p(9) + p(7)) * 4.0
        print("  Fired Anger AU16+AU25/27+AU9+AU7 rule")
    if p(16) > 0.9 and p(10) > 0.85 and (p(25) > 0.8 or p(26) > 0.8 or p(27) > 0.7) and p(1) < 0.45:
        scores['Anger'] += (p(16) + p(10) + max(p(25), p(26), p(27))) * 5.5
        print("  Fired Anger AU16+AU10 rule")

    return scores

for item in data:
    if item['img'] in ['test_1791_aligned.jpg', 'test_1301_aligned.jpg', 'test_2099_aligned.jpg']:
        print("===", item['img'], "===")
        s = debug_v8(item['au_probs'])
        print("Scores:", s)
