import pickle, re

with open('scratch/data_35.pkl', 'rb') as f:
    data = pickle.load(f)

def canonical_au(raw_name):
    m = re.match(r'^(?:[LRBTt]{1,2}|AD|BL|RB)?(\d+)[A-Za-z]?$', str(raw_name))
    if m:
        return m.group(1)
    return str(raw_name)

def classify_optimized(au_probs):
    c_probs = {}
    for au, pr in au_probs.items():
        c = canonical_au(au)
        c_probs[c] = max(c_probs.get(c, 0.0), float(pr))

    def p(c): return c_probs.get(str(c), 0.0)

    l12 = au_probs.get('L12', 0.0)
    r12 = au_probs.get('R12', 0.0)
    l14 = au_probs.get('L14', 0.0)
    r14 = au_probs.get('R14', 0.0)
    is_unilateral_smile = (l12 > 0.4 and r12 < 0.25) or (r12 > 0.4 and l12 < 0.25) or (l12 > 0.55 and r12 < 0.35) or (r12 > 0.55 and l12 < 0.35)
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

    # ── 1. SURPRISE ──
    # Arched high brows (1+2), wide eyes (5), slack/open jaw (26/27)
    if p(1) > 0.4 and p(2) > 0.4 and p(5) > 0.4:
        scores['Surprise'] += (p(1) + p(2) + p(5)) * 2.2
        if p(26) > 0.4 or p(27) > 0.4:
            scores['Surprise'] += max(p(26), p(27)) * 2.5
    elif p(5) > 0.5 and (p(26) > 0.4 or p(27) > 0.4) and p(16) < 0.4:
        scores['Surprise'] += (p(5) + max(p(26), p(27))) * 2.2
        if p(1) > 0.4 or p(2) > 0.4:
            scores['Surprise'] += (p(1) + p(2)) * 1.5
    elif p(27) > 0.7 and p(16) < 0.4 and p(4) < 0.4 and p(9) < 0.4:
        scores['Surprise'] += p(27) * 3.5
    elif (p(1) > 0.5 and p(2) > 0.5) and (p(26) > 0.5 or p(27) > 0.5) and p(16) < 0.4:
        scores['Surprise'] += (p(1) + p(2)) * 2.0
    if p(19) > 0.5:
        scores['Surprise'] *= 0.1

    # ── 2. FEAR ──
    # Hallmark: Mouth stretch (27) + lower teeth bare (16)
    # If brow furrow (4) is moderate/low (<0.50) -> FEAR!
    # If Risorius (20) is active with mouth open -> FEAR!
    if p(27) > 0.7 and p(16) > 0.6:
        if p(4) < 0.50:
            scores['Fear'] += (p(27) + p(16)) * 5.0
            if p(1) > 0.4 or p(2) > 0.4:
                scores['Fear'] += 2.0
            if p(20) > 0.5:
                scores['Fear'] += p(20) * 2.5
        elif p(4) >= 0.50:
            if p(4) > 0.75 and p(7) > 0.5 and p(20) < 0.4:
                scores['Anger'] += (p(27) + p(16) + p(4)) * 3.5
            else:
                scores['Fear'] += (p(27) + p(16)) * 4.5
    elif p(20) > 0.55 and (p(27) > 0.6 or p(25) > 0.6):
        scores['Fear'] += (p(20) + max(p(27), p(25))) * 4.0
        if p(5) > 0.4:
            scores['Fear'] += p(5) * 1.5
    elif (p(1) > 0.5 and p(2) > 0.5) and p(27) > 0.7:
        scores['Fear'] += (p(1) + p(2) + p(27)) * 2.5
    elif p(20) > 0.6 and p(16) > 0.5:
        scores['Fear'] += (p(20) + p(16)) * 3.5

    # ── 3. HAPPINESS ──
    # Duchenne smile: AU12 (lip corner puller) + AU6 (cheek raiser)
    # BUT if deep brow furrow (4) or nose wrinkler (9) or terror scream (27+16): NOT Happiness!
    if p(12) > 0.45:
        base_hap = p(12) * 3.0
        if p(6) > 0.4:
            base_hap += p(6) * 4.0
        if p(25) > 0.5:
            base_hap += p(25) * 0.5
        
        # Penalties:
        if p(27) > 0.7 and p(16) > 0.6: # Fear scream
            base_hap = 0.0
        elif p(4) > 0.6 and p(9) > 0.6: # Disgust sneer
            base_hap *= 0.1
        elif p(1) > 0.5 and p(4) > 0.5 and p(16) > 0.5: # Crying weeping face
            base_hap = 0.0
        elif p(14) > 0.7 and p(6) < 0.5 and (p(23) > 0.4 or p(4) > 0.4): # Contempt smirk
            base_hap *= 0.2
        elif p(16) > 0.7 and p(6) < 0.5:
            base_hap *= 0.2
        scores['Happiness'] += base_hap

    # ── 4. SADNESS ──
    # Oblique brow knot: AU1 + AU4
    # Weeping mouth: AU15 (lip corner depressor) + AU17 (chin raiser)
    # Crying face: AU1 + AU4 + AU16
    if p(1) > 0.4 and p(4) > 0.4 and p(27) < 0.6:
        scores['Sadness'] += (p(1) + p(4)) * 3.5
        if p(15) > 0.4 or p(17) > 0.4:
            scores['Sadness'] += 1.5
    if p(1) > 0.5 and p(4) > 0.5 and p(16) > 0.6: # Intense crying / weeping
        scores['Sadness'] += (p(1) + p(4) + p(16)) * 4.0
    if p(15) > 0.5 and p(17) > 0.5 and p(27) < 0.6 and p(1) > 0.35: # Weeping grief (requires brow action AU1)
        scores['Sadness'] += (p(15) + p(17)) * 3.5
        if p(4) > 0.4:
            scores['Sadness'] += p(4) * 2.0
    elif p(4) > 0.5 and p(43) > 0.5 and p(12) < 0.4 and p(23) < 0.4 and p(24) < 0.4:
        scores['Sadness'] += (p(4) + p(43)) * 2.5
    if p(12) > 0.5 and p(1) < 0.4:
        scores['Sadness'] *= 0.1
    if p(27) > 0.7:
        scores['Sadness'] *= 0.1

    # ── 5. DISGUST ──
    # Nose wrinkler (9), upper lip raiser (10), tongue show (19)
    # Sneer: AU4 + AU9 or AU4 + AU10
    if p(19) > 0.5 and p(14) < 0.6:
        scores['Disgust'] += p(19) * 8.0
    if p(9) > 0.5:
        scores['Disgust'] += p(9) * 4.0
        if p(4) > 0.4:
            scores['Disgust'] += p(4) * 3.0
    if p(10) > 0.45:
        scores['Disgust'] += p(10) * 2.5
        if p(4) > 0.4:
            scores['Disgust'] += p(4) * 2.0
    if p(4) > 0.6 and p(14) > 0.6 and p(12) > 0.6 and p(9) > 0.6:
        # Intense disgust sneer (1627)
        scores['Disgust'] += 12.0
    if p(4) > 0.4 and p(15) > 0.35 and p(17) > 0.35 and p(1) < 0.3 and p(23) < 0.3:
        scores['Disgust'] += (p(4) + p(15) + p(17)) * 3.0
    # Genuine smile suppresses disgust ONLY IF brow furrow AU4 is absent
    if p(12) > 0.8 and p(6) > 0.8 and p(4) < 0.5:
        scores['Disgust'] *= 0.2

    # ── 6. ANGER ──
    # Brow furrow (4) + lip tightener/pressor (23/24)
    # Glaring brow furrow without grief (AU4 > 0.6, AU1 < 0.35, AU15/17 > 0.6) -> Pouting anger!
    if p(4) > 0.4 and (p(23) > 0.4 or p(24) > 0.4):
        scores['Anger'] += (p(4) + max(p(23), p(24))) * 4.0
        if p(15) > 0.4:
            scores['Anger'] += p(15) * 1.5
    if p(4) > 0.6 and p(1) < 0.35 and (p(15) > 0.6 or p(17) > 0.6):
        # Pouting / sulking anger (AU4 + AU15/17 with flat brow AU1<0.35)
        scores['Anger'] += (p(4) + max(p(15), p(17))) * 3.5
    if p(16) > 0.7 and (p(25) > 0.7 or p(27) > 0.7) and (p(9) > 0.6 or p(10) > 0.6) and p(4) > 0.5:
        scores['Anger'] += (p(16) + max(p(25), p(27)) + max(p(9), p(10))) * 3.0
    if p(4) > 0.6 and p(7) > 0.5:
        scores['Anger'] += (p(4) + p(7)) * 2.0
    if p(12) > 0.6 and p(6) > 0.5 and p(4) < 0.4:
        scores['Anger'] *= 0.1

    # ── 7. CONTEMPT ──
    # Unilateral smirk (L12 or L14)
    # Dimpler (14) + lip tightener (23) or brow furrow (4)
    if is_unilateral_smile or is_unilateral_dimple:
        scores['Contempt'] += 6.0
    if p(14) > 0.5:
        if (p(23) > 0.4 or p(24) > 0.4 or p(4) > 0.4) and p(6) < 0.7:
            scores['Contempt'] += (p(14) + max(p(23), p(24), p(4))) * 3.0
        if p(12) > 0.5 and p(6) < 0.7:
            scores['Contempt'] += (p(14) + p(12)) * 2.5
    if p(24) > 0.6 and p(23) > 0.6 and p(14) > 0.5:
        scores['Contempt'] += (p(24) + p(23) + p(14)) * 2.5

    best = max(scores, key=scores.get)
    if scores[best] < 0.4:
        return 'Neutral / Unknown', scores
    return best, scores

correct = 0
for item in data:
    pred, sc = classify_optimized(item['au_probs'])
    m = (pred == item['gt'])
    if m: correct += 1
    m_str = "CORRECT" if m else "MISMATCH"
    print(f"[{item['folder']}] {item['img']:25s} | GT: {item['gt']:10s} | Pred: {pred:10s} | {m_str}")

acc = correct / len(data) * 100
print(f"\n>>> Total Accuracy: {correct}/{len(data)} ({acc:.1f}%) <<<")
