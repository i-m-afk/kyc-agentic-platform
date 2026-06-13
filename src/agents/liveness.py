import os
from src.schemas.models import LivenessResult, LivenessStatus
from src.utils.helpers import get_mock_ml_flag, get_liveness_model_path

# Guarded imports for optional heavy libraries
OPENCV_AVAILABLE = False
try:
    import cv2
    OPENCV_AVAILABLE = True
except ImportError:
    pass

TORCH_AVAILABLE = False
try:
    import torch
    import torch.nn as nn
    from torchvision import models, transforms
    from PIL import Image
    TORCH_AVAILABLE = True
except ImportError:
    pass

# Keep LivenessModel definition local to support actual PyTorch inference loading
if TORCH_AVAILABLE:
    class LivenessModel(nn.Module):
        def __init__(self):
            super(LivenessModel, self).__init__()
            self.backbone = models.mobilenet_v3_small(weights=None) # We load custom weights
            in_features = self.backbone.classifier[3].in_features
            self.backbone.classifier[3] = nn.Linear(in_features, 2)

        def forward(self, x):
            return self.backbone(x)
else:
    class LivenessModel:
        pass

def extract_frames(video_path: str, num_frames: int = 5):
    """Helper to extract evenly spaced frames from a video using OpenCV."""
    if not OPENCV_AVAILABLE:
        raise ImportError("OpenCV (opencv-python-headless) is required for real video frame extraction.")
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Failed to open video file: {video_path}")
        
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        total_frames = 100 # Fallback
        
    interval = max(total_frames // num_frames, 1)
    frames = []
    
    for i in range(num_frames):
        cap.set(cv2.CAP_PROP_POS_FRAMES, min(i * interval, total_frames - 1))
        ret, frame = cap.read()
        if not ret:
            break
        # Convert BGR (OpenCV default) to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame_rgb)
        
    cap.release()
    return frames

def verify_liveness(video_path: str) -> LivenessResult:
    """
    Performs liveness check on face video.
    Supports local Mock mode and PyTorch classification.
    """
    # 1. Check if we should use local Mock mode
    if get_mock_ml_flag():
        filename = video_path.lower()
        if any(term in filename for term in ["spoof", "fail", "imposter"]):
            return LivenessResult(
                liveness_status=LivenessStatus.FAILED,
                confidence=0.92,
                spoof_probability=0.88,
                flags=["no_blink_detected", "device_screen_glare"]
            )
        else:
            return LivenessResult(
                liveness_status=LivenessStatus.PASSED,
                confidence=0.97,
                spoof_probability=0.03,
                flags=[]
            )

    # 2. Real inference logic
    if not TORCH_AVAILABLE:
        raise ImportError(
            "PyTorch and torchvision are required for real liveness model inference. "
            "Please run in Mock Mode (MOCK_ML=true) or install these packages."
        )

    model_path = get_liveness_model_path()
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Liveness model weights not found at {model_path}. "
            "Please train the model first by running notebooks/train_liveness_model.ipynb "
            "on your GPU server, or use MOCK_ML=true to mock liveness."
        )

    # Load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LivenessModel()
    
    try:
        # Load state dict
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval()
    except Exception as e:
        raise RuntimeError(f"Failed to load liveness model weights: {str(e)}")

    # Extract and preprocess frames
    try:
        frames = extract_frames(video_path, num_frames=5)
    except Exception as e:
        raise ValueError(f"Frame extraction failed: {str(e)}")

    if not frames:
        raise ValueError("Could not extract any valid frames from the video.")

    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    spoof_probs = []
    with torch.no_grad():
        for frame in frames:
            img_tensor = transform(frame).unsqueeze(0).to(device)
            outputs = model(img_tensor)
            probs = torch.softmax(outputs, dim=1)
            # Class 0: Real, Class 1: Spoof
            spoof_probs.append(probs[0][1].item())

    # Average probabilities
    avg_spoof_prob = sum(spoof_probs) / len(spoof_probs)
    passed = avg_spoof_prob < 0.5
    confidence = 1.0 - avg_spoof_prob if passed else avg_spoof_prob

    status = LivenessStatus.PASSED if passed else LivenessStatus.FAILED
    flags = []
    if not passed:
        # Generate some descriptive dummy flags if it fails spoof check
        flags.append("high_spoof_probability_texture")
        if avg_spoof_prob > 0.8:
            flags.append("device_screen_rebound")

    return LivenessResult(
        liveness_status=status,
        confidence=round(confidence, 3),
        spoof_probability=round(avg_spoof_prob, 3),
        flags=flags
    )
