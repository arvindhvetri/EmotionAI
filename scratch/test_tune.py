import pickle, re, sys

with open('scratch/data_35.pkl', 'rb') as f:
    bench35 = pickle.load(f)

with open('scratch/data_rep4.pkl', 'rb') as f:
    rep4 = pickle.load(f)

def canonical_au(raw_name):
    m = re.match(r'^(?:[LRBTt]{1,2}|AD|BL|RB)?(\d+)[A-Za-z]?$', str(raw_name))
    if m:
        return m.group(1)
    return str(raw_name)

def classify_robust(au_probs):
    c_probs = {}
    for au, pr in au_probs.items():
        c = canonical_au(au)
        c_probs[c] = max(c_probs.get(c, 0.0), float(pr))

    def p(c): return c_probs.get(str(c), 0.0)

    l12 = au_probs.get('L12', 0.0)
    r12 = au_probs.get('R12', 0.0)
    l14 = au_probs.get('L14', 0.0)
    r14 = au_probs.get('R14', 0.0)
    
    # Unilateral asymmetry must have significant ratio and low bilateral counterpart
    is_bilateral_smile = (l12 > 0.45 and r12 > 0.45) or (p(12) > 0.6 and p(6) > 0.4)
    is_unilateral_smile = (abs(l12 - r12) > 0.35) and (max(l12, r12) > 0.45) and not is_bilateral_smile
    is_unilateral_dimple = (abs(l14 - r14) > 0.35) and (max(l14, r14) > 0.45) and not is_bilateral_smile

    scores = {
        'Surprise': 0.1,
        'Fear': 0.1,
        'Disgust': 0.1,
        'Happiness': 0.1,
        'Sadness': 0.1,
        'Anger': 0.1,
        'Contempt': 0.1
    }

    # 1. SURPRISE: Arched brows (1+2), wide eyes (5), open mouth / jaw drop (26/27)
    if p(1) > 0.35 and p(2) > 0.35:
        scores['Surprise'] += (p(1) + p(2)) * 2.5
        if p(5) > 0.35: scores['Surprise'] += p(5) * 2.5
        if p(26) > 0.35 or p(27) > 0.35: scores['Surprise'] += max(p(26), p(27)) * 2.5
    elif p(5) > 0.4 and (p(26) > 0.4 or p(27) > 0.4) and p(16) < 0.4:
        scores['Surprise'] += (p(5) + max(p(26), p(27))) * 2.2
        if p(1) > 0.35 or p(2) > 0.35: scores['Surprise'] += 2.0
    elif p(27) > 0.6 and p(16) < 0.4 and p(4) < 0.4 and p(9) < 0.4:
        scores['Surprise'] += p(27) * 3.5
    if p(19) > 0.5:
        scores['Surprise'] *= 0.1

    # 2. FEAR: AU20 lip stretcher, AU1+AU2+AU4+AU5, terror scream AU27+AU16+AU20
    if p(20) > 0.4:
        scores['Fear'] += p(20) * 4.5
        if p(1) > 0.3 or p(2) > 0.3: scores['Fear'] += 2.0
        if p(5) > 0.3: scores['Fear'] += p(5) * 2.0
        if p(27) > 0.5: scores['Fear'] += p(27) * 2.5
    if p(27) > 0.6 and p(16) > 0.5:
        if p(20) > 0.4 or p(1) > 0.4 or p(2) > 0.4 or p(12) > 0.6:
            scores['Fear'] += (p(27) + p(16)) * 4.0
        elif p(4) > 0.5 and (p(7) > 0.5 or p(9) > 0.5 or p(10) > 0.5):
            scores['Anger'] += (p(27) + p(16) + p(4)) * 3.5
        else:
            scores['Fear'] += (p(27) + p(16)) * 3.0
    if p(1) > 0.4 and p(2) > 0.4 and p(4) > 0.4: # Fear brow
        scores['Fear'] += (p(1) + p(2) + p(4)) * 2.5

    # 3. HAPPINESS: Duchenne smile AU12 + AU6
    if p(12) > 0.4:
        base_hap = p(12) * 3.5
        if p(6) > 0.35: base_hap += p(6) * 4.5
        if p(25) > 0.35: base_hap += p(25) * 1.5
        # Fear/Anger grimace override
        if p(27) > 0.7 and p(16) > 0.6 and p(6) < 0.5:
            base_hap = 0.0
        elif p(1) > 0.5 and p(4) > 0.5 and p(16) > 0.5: # Weeping
            base_hap = 0.0
        scores['Happiness'] += base_hap

    # 4. SADNESS: AU1 + AU4 medial brow triangle, AU15 lip depressor, AU17 chin raiser
    if p(1) > 0.35 and p(4) > 0.35 and p(27) < 0.6:
        scores['Sadness'] += (p(1) + p(4)) * 4.5
        if p(15) > 0.35: scores['Sadness'] += p(15) * 2.5
        if p(17) > 0.35: scores['Sadness'] += p(17) * 2.0
    if p(15) > 0.4 and p(17) > 0.4:
        scores['Sadness'] += (p(15) + p(17)) * 3.0
        if p(4) > 0.35: scores['Sadness'] += p(4) * 2.0
    elif p(4) > 0.4 and p(43) > 0.4 and p(12) < 0.4:
        scores['Sadness'] += (p(4) + p(43)) * 2.5
    if p(12) > 0.7 and not (p(1) > 0.5 and p(4) > 0.5):
        scores['Sadness'] *= 0.1

    # 5. DISGUST: AU9 nose wrinkler, AU10 upper lip raiser, AU15+AU17 grimace
    if p(19) > 0.4: scores['Disgust'] += p(19) * 7.0
    if p(9) > 0.4:
        scores['Disgust'] += p(9) * 4.5
        if p(4) > 0.35: scores['Disgust'] += p(4) * 2.5
    if p(10) > 0.4:
        scores['Disgust'] += p(10) * 3.5
        if p(4) > 0.35: scores['Disgust'] += p(4) * 2.0
    if p(15) > 0.5 and p(17) > 0.5 and p(4) > 0.5 and p(1) < 0.4:
        scores['Disgust'] += (p(15) + p(17) + p(4)) * 2.5
    # Duchenne laugh suppresses disgust
    if p(12) > 0.75 and p(6) > 0.7 and p(20) < 0.5:
        scores['Disgust'] *= 0.1

    # 6. ANGER: AU4 brow lowerer + AU23/AU24 lip tightener/pressor, AU16+AU25+AU7 roaring
    if p(4) > 0.4 and (p(23) > 0.35 or p(24) > 0.35):
        scores['Anger'] += (p(4) + max(p(23), p(24))) * 4.0
        if p(17) > 0.35: scores['Anger'] += p(17) * 2.0
    if p(4) > 0.5 and p(1) < 0.4 and (p(15) > 0.5 or p(17) > 0.5):
        scores['Anger'] += (p(4) + max(p(15), p(17))) * 4.0
    if p(16) > 0.6 and (p(25) > 0.6 or p(27) > 0.6) and (p(7) > 0.5 or p(9) > 0.5 or p(10) > 0.5):
        scores['Anger'] += (p(16) + max(p(25), p(27))) * 3.5
    if p(12) > 0.6 and p(6) > 0.5 and p(4) < 0.4:
        scores['Anger'] *= 0.1

    # 7. CONTEMPT: Unilateral smirk / dimple, or tight-lipped smirk AU14 + AU23/AU24
    if not is_bilateral_smile and (is_unilateral_smile or is_unilateral_dimple):
        scores['Contempt'] += 4.5
    if p(14) > 0.5 and not is_bilateral_smile:
        if (p(23) > 0.4 or p(24) > 0.4) and p(15) < 0.4 and p(17) < 0.4:
            scores['Contempt'] += (p(14) + max(p(23), p(24))) * 3.5

    best = max(scores, key=scores.get)
    return best

def eval_set(name, items):
    corr = 0
    for it in items:
        pred = classify_robust(it['au_probs'])
        if pred == it['gt']: corr += 1
    pct = corr / len(items) * 100
    print(f"Accuracy on {name}: {corr}/{len(items)} ({pct:.1f}%)")

eval_set("Original Benchmark 35", bench35)
eval_set("Report 4 (Random 35)", rep4)
