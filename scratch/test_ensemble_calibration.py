import os, sys, random, torch
sys.path.insert(0, r'd:\Research4')
import numpy as np
from PIL import Image
from torchvision import transforms
from facenet_pytorch import InceptionResnetV1
import app

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Model 1: Emotion model
m_em = InceptionResnetV1(pretrained=None, classify=True, num_classes=7).to(device)
m_em.load_state_dict(torch.load('models/best_vggface2_emotion.pth', map_location=device))
m_em.eval()

# Model 2: AU model
m_au = app.model
all_aus = app.all_aus
m_au.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

em_map = {0: 'Surprise', 1: 'Fear', 2: 'Disgust', 3: 'Happiness', 4: 'Sadness', 5: 'Anger', 6: 'Contempt'}
test_dir = r'D:\Research4\TestDb\DATASET\test'

def predict_calibrated(image):
    t = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        em_logits = m_em(t)[0]
        au_probs_t = torch.sigmoid(m_au(t))[0]
    
    probs = torch.softmax(em_logits, dim=0).cpu().numpy()
    au_p = {all_aus[k]: float(au_probs_t[k]) for k in range(len(all_aus))}
    
    def p(c): return max([au_p.get(k, 0.0) for k in [str(c), f'L{c}', f'R{c}', f'B{c}', f'T{c}'] if k in au_p] + [0.0])
    
    scores = np.copy(probs)
    
    # Synergistic calibration between Action Units and Emotion Probabilities:
    # 0: Surprise: AU1+AU2+AU5 + (AU26 or AU27)
    if (p(1) > 0.3 or p(2) > 0.3) and (p(26) > 0.35 or p(27) > 0.35) and p(4) < 0.4:
        scores[0] *= 1.8
    if p(27) > 0.6 and p(4) < 0.4:
        scores[0] *= 1.6
        
    # 1: Fear: AU1+AU2 + AU4 + AU20
    if p(20) > 0.35 and (p(1) > 0.3 or p(4) > 0.3):
        scores[1] *= 2.2
    if p(20) > 0.5:
        scores[1] *= 1.8
        
    # 2: Disgust: AU9 or AU10
    if p(9) > 0.35 or p(10) > 0.35:
        scores[2] *= 2.0
    if p(9) > 0.5:
        scores[2] *= 1.8
        
    # 3: Happiness: AU6 + AU12
    if p(12) > 0.4:
        scores[3] *= 1.5
    if p(6) > 0.4 and p(12) > 0.4:
        scores[3] *= 1.8
        
    # 4: Sadness: AU1 + AU4 + AU15 / AU17
    if p(4) > 0.35 and (p(15) > 0.3 or p(17) > 0.3):
        scores[4] *= 1.9
    if p(15) > 0.4:
        scores[4] *= 1.7
        
    # 5: Anger: AU4 + (AU23 or AU24)
    if p(4) > 0.4 and (p(23) > 0.35 or p(24) > 0.35):
        scores[5] *= 1.9
    if p(4) > 0.6 and p(7) > 0.4:
        scores[5] *= 1.6
        
    # 6: Contempt: unilateral 12 or 14
    l12 = au_p.get('L12', 0.0)
    r12 = au_p.get('R12', 0.0)
    l14 = au_p.get('L14', 0.0)
    r14 = au_p.get('R14', 0.0)
    if ((l12 > 0.4 and r12 < 0.25) or (r12 > 0.4 and l12 < 0.25)) and p(6) < 0.5:
        scores[6] *= 1.7
    if (l14 > 0.4 and r14 < 0.25) or (r14 > 0.4 and l14 < 0.25):
        scores[6] *= 1.7
        
    return em_map[int(np.argmax(scores))]

# Test on 10 random seeds
results = []
for seed in range(1, 11):
    random.seed(seed * 77)
    correct, total = 0, 0
    for i in range(1, 8):
        true_em = em_map[i-1]
        folder = os.path.join(test_dir, str(i))
        imgs = [f for f in os.listdir(folder) if f.lower().endswith(('.jpg', '.png'))]
        sel = random.sample(imgs, 5)
        for fn in sel:
            img = Image.open(os.path.join(folder, fn)).convert('RGB')
            pred = predict_calibrated(img)
            if pred == true_em:
                correct += 1
            total += 1
    acc = correct / total
    results.append(acc)
    print(f"Seed {seed}: {correct}/{total} = {acc*100:.1f}%")

print(f"\nAverage Accuracy across 10 random sample runs: {np.mean(results)*100:.1f}%")
