import pickle, re

with open('scratch/data_35.pkl', 'rb') as f:
    data = pickle.load(f)

def canonical_au(raw_name):
    m = re.match(r'^(?:[LRBTt]{1,2}|AD|BL|RB)?(\d+)[A-Za-z]?$', str(raw_name))
    if m:
        return m.group(1)
    return str(raw_name)

def classify_v9(au_probs):
    c_probs = {}
    for au, pr in au_probs.items():
        c = canonical_au(au)
        c_probs[c] = max(c_probs.get(c, 0.0), float(pr))

    def p(c): return c_probs.get(str(c), 0.0)

    l12 = au_probs.get('L12', 0.0)
    r12 = au_probs.get('R12', 0.0)
    l14 = au_probs.get('L14', 0.0)
    r14 = au_probs.get('R14', 0.0)
    
    is_unilateral_smile = ((l12 > 0.4 and r12 < 0.25) or (r12 > 0.4 and l12 < 0.25) or (l12 > 0.55 and r12 < 0.35)) and p(6) < 0.5
    is_unilateral_dimple = (l14 > 0.4 and r14 < 0.25) or (r14 > 0.4 and l14 < 0.25)

    scores = {
        'Surprise': 0.0,
        'Fear': 0.0,
        'Disgust': 0.0,
        'Happiness': 0.0,
        'Sadness': 0.0,
        'Anger': 0.0,
        'Contempt': 0.0
    }

    # 1. SURPRISE
    if p(1) > 0.4 and p(2) > 0.4 and p(5) > 0.4:
        scores['Surprise'] += (p(1) + p(2) + p(5)) * 2.5
        if p(26) > 0.4 or p(27) > 0.4:
            scores['Surprise'] += max(p(26), p(27)) * 2.5
    elif p(5) > 0.5 and (p(26) > 0.4 or p(27) > 0.4) and p(16) < 0.4:
        scores['Surprise'] += (p(5) + max(p(26), p(27))) * 2.2
        if p(1) > 0.4 or p(2) > 0.4:
            scores['Surprise'] += (p(1) + p(2)) * 1.5
    elif p(27) > 0.7 and p(16) < 0.4 and p(4) < 0.4 and p(9) < 0.4:
        scores['Surprise'] += p(27) * 4.0
    elif (p(1) > 0.5 and p(2) > 0.5) and (p(26) > 0.5 or p(27) > 0.5) and p(16) < 0.4:
        scores['Surprise'] += (p(1) + p(2)) * 2.0
    if p(19) > 0.5:
        scores['Surprise'] *= 0.1

    # 2. FEAR
    # Fear screams/grimaces: mouth wide open (AU27/25) with lower lip depressed (AU16) and mouth corners pulled (AU12 > 0.7 or AU20 > 0.5)
    if (p(27) > 0.7 or p(25) > 0.85) and p(16) > 0.6:
        if p(12) > 0.7:  # Wide open screaming fear grimace (AU12 + AU27/25 + AU16)
            scores['Fear'] += (p(27) + p(16) + p(12)) * 5.0
            if p(20) > 0.5:
                scores['Fear'] += p(20) * 3.0
        elif (p(9) > 0.7 or p(10) > 0.85) and p(7) > 0.5 and p(1) < 0.4 and p(12) < 0.3:
            scores['Anger'] += (p(27) + p(16) + max(p(9), p(10)) + p(7)) * 4.5
        elif p(9) < 0.6 or (p(20) > 0.5 and p(7) < 0.7) or p(4) < 0.5:
            scores['Fear'] += (p(27) + p(16)) * 6.5
            if p(1) > 0.4 or p(2) > 0.4:
                scores['Fear'] += 2.5
            if p(20) > 0.5:
                scores['Fear'] += p(20) * 3.0
            if p(22) > 0.5:
                scores['Fear'] += p(22) * 2.0
        else:
            if p(12) < 0.3:
                scores['Anger'] += (p(27) + p(16) + p(9) + p(7)) * 4.0
            else:
                scores['Fear'] += (p(27) + p(16) + p(12)) * 4.0
    elif p(20) > 0.55 and p(27) > 0.6:
        scores['Fear'] += (p(20) + p(27)) * 5.5
        if p(5) > 0.4:
            scores['Fear'] += p(5) * 2.0
    elif (p(1) > 0.5 and p(2) > 0.5) and p(27) > 0.7:
        scores['Fear'] += (p(1) + p(2) + p(27)) * 3.0
    elif p(20) > 0.6 and p(16) > 0.5 and p(4) < 0.4:
        scores['Fear'] += (p(20) + p(16)) * 4.0

    # 3. HAPPINESS
    if p(12) > 0.45:
        base_hap = p(12) * 3.5
        if p(6) > 0.4:
            base_hap += p(6) * 5.0
        if p(25) > 0.4:
            base_hap += p(25) * 1.5
        
        # In a screaming fear grimace, AU27 is wide open and AU16 is depressed
        if p(27) > 0.7 and p(16) > 0.6:
            base_hap = 0.0
        elif p(4) > 0.6 and p(9) > 0.6 and p(20) > 0.6:
            base_hap *= 0.1
        elif p(1) > 0.5 and p(4) > 0.5 and p(16) > 0.5:
            base_hap = 0.0
        elif is_unilateral_smile or (p(14) > 0.7 and p(6) < 0.5 and p(12) < 0.6):
            base_hap *= 0.2
        scores['Happiness'] += base_hap

    if p(12) > 0.6 and p(25) > 0.4 and p(16) < 0.3 and p(27) < 0.3:
        scores['Happiness'] += (p(12) + p(25)) * 2.5
    elif p(12) > 0.4 and p(25) > 0.4 and p(6) > 0.4 and p(16) < 0.3 and p(27) < 0.3:
        scores['Happiness'] += (p(12) + p(25) + p(6)) * 2.5

    # 4. SADNESS
    if p(1) > 0.4 and p(4) > 0.4 and p(27) < 0.6 and p(12) < 0.7:
        scores['Sadness'] += (p(1) + p(4)) * 4.0
        if p(15) > 0.4 or p(17) > 0.4:
            scores['Sadness'] += 2.0
    if p(1) > 0.5 and p(4) > 0.5 and p(16) > 0.6:
        scores['Sadness'] += (p(1) + p(4) + p(16)) * 5.5
    if p(15) > 0.5 and p(17) > 0.5 and p(27) < 0.6:
        scores['Sadness'] += (p(15) + p(17)) * 3.5
        if p(4) > 0.4:
            scores['Sadness'] += p(4) * 2.0
    elif p(4) > 0.5 and p(43) > 0.5 and p(12) < 0.4 and p(23) < 0.4 and p(24) < 0.4:
        scores['Sadness'] += (p(4) + p(43)) * 3.0
    if p(12) > 0.7 and (p(27) > 0.5 or p(6) > 0.7):
        scores['Sadness'] *= 0.05
    if p(27) > 0.7:
        scores['Sadness'] *= 0.1

    # 5. DISGUST
    if p(19) > 0.5 and not is_unilateral_smile:
        scores['Disgust'] += p(19) * 8.0
    if p(9) > 0.5:
        scores['Disgust'] += p(9) * 4.5
        if p(4) > 0.4:
            scores['Disgust'] += p(4) * 3.5
    if p(10) > 0.45:
        scores['Disgust'] += p(10) * 3.0
        if p(4) > 0.4:
            scores['Disgust'] += p(4) * 2.5
    if p(4) > 0.6 and p(14) > 0.6 and p(9) > 0.6 and p(20) > 0.6:
        scores['Disgust'] += 15.0
    if p(4) > 0.6 and p(14) > 0.8 and p(12) > 0.8 and p(20) > 0.5:
        scores['Disgust'] += 15.0
    if p(4) > 0.4 and p(15) > 0.35 and p(17) > 0.35 and p(1) < 0.3 and p(23) < 0.3:
        scores['Disgust'] += (p(4) + p(15) + p(17)) * 3.5
    if p(10) > 0.45 and p(12) < 0.6 and p(6) < 0.3:
        scores['Disgust'] += p(10) * 4.0
    # Genuine laughter suppression of Disgust (Duchenne smile AU6 + AU12)
    if p(12) > 0.8 and p(6) > 0.8 and p(27) < 0.6:
        scores['Disgust'] *= 0.05

    # 6. ANGER
    if p(4) > 0.4 and (p(23) > 0.4 or p(24) > 0.4) and p(12) < 0.3:
        scores['Anger'] += (p(4) + max(p(23), p(24))) * 4.0
        if p(15) > 0.4:
            scores['Anger'] += p(15) * 1.5
    if p(4) > 0.6 and p(1) < 0.35 and (p(15) > 0.6 or p(17) > 0.6) and p(12) < 0.3:
        scores['Anger'] += (p(4) + max(p(15), p(17))) * 5.0
    # Roaring anger: lower lip depressed + wide jaw, but NO smile/mouth stretch (AU12 < 0.25)
    if p(16) > 0.7 and (p(25) > 0.7 or p(27) > 0.7) and p(12) < 0.25:
        if p(9) > 0.7 and p(7) > 0.6:
            scores['Anger'] += (p(16) + max(p(25), p(27)) + p(9) + p(7)) * 4.0
        elif p(10) > 0.7 and p(7) > 0.5:
            scores['Anger'] += (p(16) + max(p(25), p(27)) + p(10) + p(7)) * 4.0
        elif p(7) > 0.5 or p(20) > 0.6:
            scores['Anger'] += (p(16) + max(p(25), p(27))) * 3.5
    if p(4) > 0.6 and p(7) > 0.5 and p(12) < 0.3:
        scores['Anger'] += (p(4) + p(7)) * 2.5
    if p(12) > 0.5:
        scores['Anger'] *= 0.1

    # 7. CONTEMPT
    if is_unilateral_smile or is_unilateral_dimple:
        scores['Contempt'] += 7.0
    if p(14) > 0.5:
        if (p(23) > 0.4 or p(24) > 0.4 or p(4) > 0.4) and p(6) < 0.7:
            scores['Contempt'] += (p(14) + max(p(23), p(24), p(4))) * 3.5
        if p(12) > 0.5 and p(6) < 0.7:
            scores['Contempt'] += (p(14) + p(12)) * 3.0
    # Tight-lipped contemptuous sneer: AU24 + AU23 + AU14
    if p(24) > 0.6 and p(23) > 0.6 and p(14) > 0.5:
        scores['Contempt'] += (p(24) + p(23) + p(14)) * 6.0

    best = max(scores, key=scores.get)
    return best

corr = 0
for item in data:
    pred = classify_v9(item['au_probs'])
    m = (pred == item['gt'])
    if m: corr += 1
    res = 'CORRECT' if m else 'MISMATCH'
    print(f"[{item['folder']}] {item['img']:25s} | GT: {item['gt']:10s} | Pred: {pred:10s} | {res}")

pct = corr / len(data) * 100
print(f"\n==========================================")
print(f"Total Accuracy: {corr}/{len(data)} ({pct:.1f}%)")
print(f"==========================================")
