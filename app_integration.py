"""
Drop-in changes for app.py so the web app uses the new ResNet-50 + VGGFace2 emotion model.
Copy train_emotion_resnet50_vggface2.py next to app.py first.
"""
import torch
from torchvision import transforms
from train_emotion_resnet50_vggface2 import VGGFace2ResNet50, ToVGGFace2Tensor

# 1) Load the emotion engine (replaces the InceptionResnetV1 model_emotion block)
ckpt = torch.load("models/best_resnet50_vggface2_emotion.pth", map_location=device)
model_emotion = VGGFace2ResNet50(num_classes=7)
model_emotion.load_state_dict(ckpt["state_dict"])
model_emotion = model_emotion.to(device).eval()

# 2) Use the SAME preprocessing the model was trained with (BGR, 0-255, minus VGGFace2 mean).
#    Keep the old ImageNet `transform` only for the existing AU model until it is retrained.
emotion_transform = transforms.Compose([transforms.Resize((ckpt["img_size"], ckpt["img_size"])), ToVGGFace2Tensor()])

# 3) Class order is folder 1..7 -> index 0..6
EMOTION_NAMES = ckpt["classes"]   # ["Surprise","Fear","Disgust","Happiness","Sadness","Anger","Neutral"]

# 4) Inside run_inference(), replace the emotion block with model-only prediction
#    (no override_emotion, no TEST_DB_GROUND_TRUTH lookup):
#
#    em_tensor = emotion_transform(image).unsqueeze(0).to(device)
#    with torch.no_grad():
#        logits = model_emotion(em_tensor) + model_emotion(torch.flip(em_tensor, dims=[3]))
#        probs = torch.softmax(logits, dim=1)[0]
#    predicted_emotion = EMOTION_NAMES[int(probs.argmax())]
#    confidence = float(probs.max())
#
# 5) Grad-CAM for the emotion decision:
#    from pytorch_grad_cam import GradCAM
#    emotion_cam = GradCAM(model=model_emotion, target_layers=[model_emotion.layer4])
