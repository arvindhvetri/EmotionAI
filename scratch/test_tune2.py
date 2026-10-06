import pickle, re

with open('scratch/data_rep4.pkl', 'rb') as f:
    rep4 = pickle.load(f)

with open('scratch/data_35.pkl', 'rb') as f:
    bench35 = pickle.load(f)

def canonical_au(raw_name):
    m = re.match(r'^(?:[LRBTt]{1,2}|AD|BL|RB)?(\d+)[A-Za-z]?$', str(raw_name))
    if m:
        return m.group(1)
    return str(raw_name)

def classify_v16(au_probs):
    c_probs = {}
    for au, pr in au_probs.items():
        c = canonical_au(au)
        c_probs[c] = max(c_probs.get(c, 0.0), float(pr))

    def p(c): return c_probs.get(str(c), 0.0)

    l12 = au_probs.get('L12', 0.0)
    r12 = au_probs.get('R12', 0.0)
    l14 = au_probs.get('L14', 0.0)
    r14 = au_probs.get('R14', 0.0)

    # Base scores start slightly positive to avoid ties
    scores = {
        'Surprise': 0.0,
        'Fear': 0.0,
        'Disgust': 0.0,
        'Happiness': 0.0,
        'Sadness': 0.0,
        'Anger': 0.0,
        'Contempt': 0.0
    }

    # 1. SURPRISE: Arched brows (1+2), wide eyes (5), open mouth / jaw drop (26/27)
    scores['Surprise'] += (p(1) + p(2)) * 1.5 + p(5) * 2.0 + max(p(26), p(27)) * 2.0
    if p(1) > 0.3 and p(2) > 0.3 and (p(26) > 0.3 or p(27) > 0.3):
        scores['Surprise'] += 3.0
    if p(16) > 0.6 or p(4) > 0.5 or p(9) > 0.5: # Screaming fear/anger, not surprise
        scores['Surprise'] *= 0.3

    # 2. FEAR: AU20 (lip stretcher), AU1+AU2+AU4+AU5, AU27+AU16
    scores['Fear'] += p(20) * 4.0 + (p(1) + p(4)) * 1.0 + p(5) * 1.5
    if p(27) > 0.5 and p(16) > 0.4:
        scores['Fear'] += (p(27) + p(16)) * 2.5
        if p(20) > 0.3 or p(1) > 0.3:
            scores['Fear'] += 3.0
    if p(20) > 0.4 and (p(1) > 0.3 or p(5) > 0.3 or p(27) > 0.4):
        scores['Fear'] += 3.5

    # 3. HAPPINESS: Duchenne smile AU12 + AU6 + AU25
    scores['Happiness'] += p(12) * 3.5 + p(6) * 3.5 + p(25) * 1.0
    if p(12) > 0.4 and p(6) > 0.3:
        scores['Happiness'] += 4.0
    # Suppression if crying or roaring
    if p(27) > 0.6 and p(16) > 0.6 and p(6) < 0.4:
        scores['Happiness'] = 0.0
    elif p(1) > 0.4 and p(4) > 0.4 and p(16) > 0.5:
        scores['Happiness'] *= 0.1

    # 4. SADNESS: AU1+AU4 medial brow triangle, AU15 lip depressor, AU17 chin raiser, AU43
    scores['Sadness'] += (p(1) + p(4)) * 2.5 + p(15) * 3.0 + p(17) * 2.0 + p(43) * 1.5
    if p(1) > 0.35 and p(4) > 0.35:
        scores['Sadness'] += 3.5
    if p(15) > 0.35 and p(17) > 0.35:
        scores['Sadness'] += 3.0
    if p(12) > 0.6 and p(6) > 0.4:
        scores['Sadness'] *= 0.1

    # 5. DISGUST: AU9 nose wrinkler, AU10 upper lip raiser, AU19 tongue show, AU15+AU17
    scores['Disgust'] += p(9) * 4.5 + p(10) * 3.5 + p(19) * 6.0
    if p(9) > 0.35 and p(4) > 0.35:
        scores['Disgust'] += 3.0
    if p(10) > 0.35 and p(4) > 0.35:
        scores['Disgust'] += 2.5
    if p(12) > 0.7 and p(6) > 0.6 and p(20) < 0.4:
        scores['Disgust'] *= 0.1

    # 6. ANGER: AU4 brow lowerer + AU23/AU24 lip tightener/pressor, AU16+AU25+AU7 roaring
    scores['Anger'] += p(4) * 2.5 + max(p(23), p(24)) * 3.0 + p(7) * 1.5
    if p(4) > 0.4 and (p(23) > 0.35 or p(24) > 0.35):
        scores['Anger'] += 4.0
    if p(16) > 0.6 and p(25) > 0.6 and (p(7) > 0.4 or p(4) > 0.4 or p(10) > 0.4):
        scores['Anger'] += 4.5
    if p(12) > 0.6 and p(6) > 0.5 and p(4) < 0.35:
        scores['Anger'] *= 0.1

    # 7. CONTEMPT: Strict unilateral smirk ONLY if NOT a bilateral smile or disgust/sadness
    is_bilateral = (l12 > 0.4 and r12 > 0.4) or (p(12) > 0.55 and p(6) > 0.35)
    has_disgust_sadness = (p(9) > 0.5 or p(15) > 0.5 or p(17) > 0.5 or p(10) > 0.5)
    
    if not is_bilateral and not has_disgust_sadness:
        diff_12 = abs(l12 - r12)
        diff_14 = abs(l14 - r14)
        if diff_12 > 0.4 and max(l12, r12) > 0.45:
            scores['Contempt'] += diff_12 * 5.0
        if diff_14 > 0.4 and max(l14, r14) > 0.45:
            scores['Contempt'] += diff_14 * 5.0
        if p(14) > 0.45 and (p(23) > 0.35 or p(24) > 0.35):
            scores['Contempt'] += p(14) * 3.0

    best = max(scores, key=scores.get)
    return best

def eval_data(name, dataset):
    corr = 0
    by_cat = {}
    for item in dataset:
        pred = classify_v16(item['au_probs'])
        gt = item['gt']
        m = (pred == gt)
        if m: corr += 1
        if gt not in by_cat: by_cat[gt] = [0, 0]
        by_cat[gt][1] += 1
        if m: by_cat[gt][0] += 1
    print(f"=== {name}: {corr}/{len(dataset)} ({corr/len(dataset)*100:.1f}%) ===")
    for k, v in sorted(by_cat.items()):
        print(f"  {k:12s}: {v[0]}/{v[1]} ({v[0]/v[1]*100:.1f}%)")

eval_data("Report 4 (Random 35)", rep4)
eval_data("Benchmark 35", bench35)
