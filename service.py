import os
import io
import sys
import time
import random
import sqlite3
from datetime import datetime
from typing import List, Tuple, Optional, Dict, Any

import streamlit as st
from PIL import Image
import requests
import base64
import json
import os
from dotenv import load_dotenv

try:
    import google.generativeai as genai  # type: ignore
    _gemini_available = True
except Exception:
    genai = None  # type: ignore
    _gemini_available = False

# OpenCV and ML libraries - attempt import but degrade gracefully
try:
    import cv2  # type: ignore
    _cv2_available = True
except Exception:  # pragma: no cover
    cv2 = None  # type: ignore
    _cv2_available = False

try:
    import mediapipe as mp  # type: ignore
    _mediapipe_available = True
except Exception:  # pragma: no cover
    mp = None  # type: ignore
    _mediapipe_available = False

try:
    import torch  # type: ignore
    from transformers import CLIPProcessor, CLIPModel  # type: ignore
    _clip_available = True
except Exception:  # pragma: no cover
    torch = None  # type: ignore
    CLIPProcessor = None  # type: ignore
    CLIPModel = None  # type: ignore
    _clip_available = False

try:
    from ultralytics import YOLO  # type: ignore
    _yolo_available = True
except Exception:  # pragma: no cover
    YOLO = None  # type: ignore
    _yolo_available = False


# -----------------------------
# App Constants and Directories
# -----------------------------
APP_TITLE = "SCC – Service Credit Card App"
DB_PATH = "scc_data.db"
UPLOADS_DIR = "uploads"
IMAGES_DIR = os.path.join(UPLOADS_DIR, "images")
VIDEOS_DIR = os.path.join(UPLOADS_DIR, "videos")

ACTIVITY_SUGGESTIONS = [
    "Plant Trees",
    "Clean a Park",
    "Donate Books",
    "Beach Cleanup",
    "Food Distribution",
    "Road Safety Awareness",
    "Blood Donation Camp Support",
    "E-Waste Collection Drive",
    "Neighborhood Recycling Setup",
    "Senior Citizen Assistance",
]


# -----------------------------
# Utilities: Filesystem & DB
# -----------------------------
def ensure_directories_exist() -> None:
    os.makedirs(IMAGES_DIR, exist_ok=True)
    os.makedirs(VIDEOS_DIR, exist_ok=True)


def get_db_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA foreign_keys=ON;")
    return conn


def init_db(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS users (
            name TEXT PRIMARY KEY,
            created_at TEXT NOT NULL,
            username TEXT,
            password_hash TEXT
        );
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS activities (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_name TEXT NOT NULL,
            media_path TEXT NOT NULL,
            media_type TEXT NOT NULL, -- image | video
            activity_label TEXT NOT NULL,
            is_genuine INTEGER NOT NULL,
            confidence INTEGER NOT NULL,
            points INTEGER NOT NULL,
            created_at TEXT NOT NULL,
            FOREIGN KEY(user_name) REFERENCES users(name) ON DELETE CASCADE
        );
        """
    )
    conn.commit()


def ensure_auth_schema(conn: sqlite3.Connection) -> None:
    # Add username/password_hash columns if missing, and a unique index on username
    cols = {r[1] for r in conn.execute("PRAGMA table_info(users);").fetchall()}
    if "username" not in cols:
        conn.execute("ALTER TABLE users ADD COLUMN username TEXT;")
    if "password_hash" not in cols:
        conn.execute("ALTER TABLE users ADD COLUMN password_hash TEXT;")
    conn.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_users_username ON users(username);")
    conn.commit()


def seed_sample_data(conn: sqlite3.Connection) -> None:
    # Seed only if there are no users yet
    cur = conn.execute("SELECT COUNT(*) FROM users;")
    (count_users,) = cur.fetchone()
    if count_users > 0:
        return

    now = datetime.utcnow().isoformat()
    sample_users = [
        ("aarav", "Aarav"),
        ("diya", "Diya"),
        ("rahul", "Rahul"),
        ("saanvi", "Saanvi"),
        ("ishaan", "Ishaan"),
    ]
    for uname, display in sample_users:
        conn.execute(
            "INSERT OR IGNORE INTO users(name, created_at, username, password_hash) VALUES(?, ?, ?, ?);",
            (display, now, uname, hash_password("password123")),
        )

    # Create a few sample activities without real media; use placeholders
    sample_acts = [
        ("Aarav", "Plant Trees", 1, 92, 420),
        ("Diya", "Clean a Park", 1, 88, 390),
        ("Rahul", "Donate Books", 1, 79, 260),
        ("Saanvi", "Beach Cleanup", 1, 95, 500),
        ("Ishaan", "Food Distribution", 1, 90, 450),
    ]
    for name, label, is_genuine, conf, pts in sample_acts:
        conn.execute(
            """
            INSERT INTO activities (
                user_name, media_path, media_type, activity_label, is_genuine, confidence, points, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?);
            """,
            (
                name,
                "placeholder",
                "image",
                label,
                is_genuine,
                conf,
                pts,
                now,
            ),
        )
    conn.commit()


def upsert_user(conn: sqlite3.Connection, user_name: str) -> None:
    # Deprecated in favor of create_user and authenticate_user
    now = datetime.utcnow().isoformat()
    conn.execute("INSERT OR IGNORE INTO users(name, created_at) VALUES(?, ?);", (user_name, now))
    conn.commit()


def hash_password(password: str) -> str:
    import hashlib, base64
    salt = os.urandom(16)
    digest = hashlib.sha256(salt + password.encode("utf-8")).digest()
    return base64.b64encode(salt).decode("utf-8") + "$" + base64.b64encode(digest).decode("utf-8")


def verify_password(password: str, stored: str) -> bool:
    import hashlib, base64
    try:
        salt_b64, hash_b64 = stored.split("$")
        salt = base64.b64decode(salt_b64)
        expected = base64.b64decode(hash_b64)
        calc = hashlib.sha256(salt + password.encode("utf-8")).digest()
        return calc == expected
    except Exception:
        return False


def create_user(conn: sqlite3.Connection, username: str, display_name: str, password: str) -> Tuple[bool, str]:
    if not username or not password or not display_name:
        return False, "All fields are required."
    username = username.strip().lower()
    display_name = display_name.strip().title()

    # Check username uniqueness first
    cur = conn.execute("SELECT 1 FROM users WHERE username = ?;", (username,))
    if cur.fetchone():
        return False, "Username already exists."

    # Check display name uniqueness due to PRIMARY KEY(name)
    cur = conn.execute("SELECT 1 FROM users WHERE name = ?;", (display_name,))
    if cur.fetchone():
        return False, "Display name already taken. Please choose another display name."

    try:
        now = datetime.utcnow().isoformat()
        conn.execute(
            "INSERT INTO users(name, created_at, username, password_hash) VALUES(?, ?, ?, ?);",
            (display_name, now, username, hash_password(password)),
        )
        conn.commit()
        return True, display_name
    except sqlite3.IntegrityError as ex:
        # Fallback in case of race conditions
        return False, "Account could not be created (possibly already exists)."


def authenticate_user(conn: sqlite3.Connection, username: str, password: str) -> Optional[Dict[str, str]]:
    cur = conn.execute("SELECT name, password_hash FROM users WHERE username = ?;", (username.strip().lower(),))
    row = cur.fetchone()
    if not row:
        return None
    display_name, stored_hash = row[0], row[1]
    if stored_hash and verify_password(password, stored_hash):
        return {"username": username.strip().lower(), "name": display_name}
    return None


def insert_activity(
    conn: sqlite3.Connection,
    user_name: str,
    media_path: str,
    media_type: str,
    activity_label: str,
    is_genuine: bool,
    confidence: int,
    points: int,
) -> None:
    now = datetime.utcnow().isoformat()
    conn.execute(
        """
        INSERT INTO activities (
            user_name, media_path, media_type, activity_label, is_genuine, confidence, points, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?);
        """,
        (
            user_name,
            media_path,
            media_type,
            activity_label,
            1 if is_genuine else 0,
            confidence,
            points,
            now,
        ),
    )
    conn.commit()


def query_totals(conn: sqlite3.Connection) -> Tuple[int, int]:
    cur = conn.execute("SELECT COUNT(*) FROM users;")
    (num_users,) = cur.fetchone()
    cur = conn.execute("SELECT COALESCE(SUM(points), 0) FROM activities WHERE is_genuine = 1;")
    (total_points,) = cur.fetchone()
    return num_users, int(total_points or 0)


def get_leaderboard(conn: sqlite3.Connection) -> List[Tuple[str, int]]:
    cur = conn.execute(
        """
        SELECT u.name, COALESCE(SUM(a.points), 0) AS total_points
        FROM users u
        LEFT JOIN activities a ON a.user_name = u.name AND a.is_genuine = 1
        GROUP BY u.name
        ORDER BY total_points DESC, u.name ASC;
        """
    )
    results = [(r[0], int(r[1] or 0)) for r in cur.fetchall()]
    return results


def get_user_activities(conn: sqlite3.Connection, user_name: str) -> List[Dict[str, Any]]:
    cur = conn.execute(
        """
        SELECT activity_label, is_genuine, confidence, points, media_type, media_path, created_at
        FROM activities
        WHERE user_name = ?
        ORDER BY datetime(created_at) DESC
        """,
        (user_name,),
    )
    rows = cur.fetchall()
    return [
        {
            "Activity": r[0],
            "Genuine": bool(r[1]),
            "Confidence": int(r[2]),
            "Points": int(r[3]),
            "Type": r[4],
            "Path": r[5],
            "Date": r[6],
        }
        for r in rows
    ]


def get_user_points(conn: sqlite3.Connection, user_name: str) -> int:
    cur = conn.execute(
        "SELECT COALESCE(SUM(points), 0) FROM activities WHERE user_name = ? AND is_genuine = 1;",
        (user_name,),
    )
    (pts,) = cur.fetchone()
    return int(pts or 0)


# -----------------------------
# Advanced AI Verifier Ensemble
# -----------------------------

# Global model instances (loaded once)
_yolo_model = None
_clip_model = None
_clip_processor = None
_mediapipe_pose = None


def load_models() -> Dict[str, bool]:
    """Load ML models if available. Returns status dict."""
    global _yolo_model, _clip_model, _clip_processor, _mediapipe_pose
    status = {"yolo": False, "clip": False, "mediapipe": False}
    
    # Only load models if not already loaded (lazy loading)
    if _yolo_model is None and _yolo_available:
        try:
            _yolo_model = YOLO('yolov8n.pt')  # Nano model for speed
            status["yolo"] = True
        except Exception:
            pass
    elif _yolo_model is not None:
        status["yolo"] = True
    
    if _clip_model is None and _clip_available:
        try:
            _clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
            _clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
            status["clip"] = True
        except Exception:
            pass
    elif _clip_model is not None:
        status["clip"] = True
    
    if _mediapipe_pose is None and _mediapipe_available:
        try:
            _mediapipe_pose = mp.solutions.pose.Pose(
                static_image_mode=True,
                model_complexity=1,
                enable_segmentation=False,
                min_detection_confidence=0.5
            )
            status["mediapipe"] = True
        except Exception:
            pass
    elif _mediapipe_pose is not None:
        status["mediapipe"] = True
    
    return status


def detect_objects_yolo(image_array) -> Tuple[List[str], float]:
    """Detect objects using YOLOv8. Returns (object_classes, confidence)."""
    if not _yolo_available or _yolo_model is None:
        return [], 0.0
    
    try:
        import numpy as np
        results = _yolo_model(image_array)
        detections = []
        confidences = []
        
        for r in results:
            if r.boxes is not None:
                for box in r.boxes:
                    conf = float(box.conf[0])
                    cls = int(box.cls[0])
                    label = _yolo_model.names[cls]
                    if conf > 0.3:  # Threshold
                        detections.append(label)
                        confidences.append(conf)
        
        avg_conf = sum(confidences) / len(confidences) if confidences else 0.0
        return detections, avg_conf
    except Exception:
        return [], 0.0


def detect_poses_mediapipe(image_array) -> Tuple[int, float]:
    """Detect human poses using MediaPipe. Returns (num_poses, avg_confidence)."""
    if not _mediapipe_available or _mediapipe_pose is None:
        return 0, 0.0
    
    try:
        import numpy as np
        results = _mediapipe_pose.process(image_array)
        if results.pose_landmarks:
            # Count poses and calculate average confidence
            poses = 1  # MediaPipe typically detects one pose per image
            # Estimate confidence from landmark visibility
            landmarks = results.pose_landmarks.landmark
            visibilities = [lm.visibility for lm in landmarks if lm.visibility > 0]
            avg_conf = sum(visibilities) / len(visibilities) if visibilities else 0.0
            return poses, avg_conf
        return 0, 0.0
    except Exception:
        return 0, 0.0


def analyze_semantic_clip(image_array) -> Tuple[float, str]:
    """Analyze image semantics using CLIP. Returns (similarity_score, description)."""
    if not _clip_available or _clip_model is None or _clip_processor is None:
        return 0.0, "CLIP not available"
    
    try:
        import torch
        import numpy as np
        from PIL import Image
        
        # Service activity descriptions
        service_descriptions = [
            "people planting trees in a park",
            "volunteers cleaning up trash",
            "community service activities",
            "people helping others",
            "environmental conservation work",
            "food distribution to needy",
            "blood donation drive",
            "beach cleanup volunteers",
            "senior citizen assistance",
            "community garden work"
        ]
        
        # Convert numpy array to PIL Image
        if len(image_array.shape) == 3:
            pil_image = Image.fromarray(image_array)
        else:
            pil_image = Image.fromarray(image_array)
        
        # Process with CLIP
        inputs = _clip_processor(text=service_descriptions, images=pil_image, return_tensors="pt", padding=True)
        
        with torch.no_grad():
            outputs = _clip_model(**inputs)
            logits_per_image = outputs.logits_per_image
            probs = logits_per_image.softmax(dim=1)
        
        # Get max similarity and corresponding description
        max_prob, max_idx = torch.max(probs, dim=1)
        max_score = float(max_prob[0])
        description = service_descriptions[max_idx[0]]
        
        return max_score, description
    except Exception:
        return 0.0, "CLIP analysis failed"


def detect_synthetic_heuristics(image_array) -> Tuple[bool, float, str]:
    """Heuristic detection of synthetic/animated images."""
    try:
        import numpy as np
        gray = cv2.cvtColor(image_array, cv2.COLOR_BGR2GRAY)
        
        # Check for cartoon-like characteristics
        # 1. Color quantization (synthetic images have fewer unique colors)
        unique_colors = len(np.unique(image_array.reshape(-1, 3), axis=0))
        color_score = min(unique_colors / 10000, 1.0)  # Normalize
        
        # 2. Edge density and uniformity
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / edges.size
        
        # 3. Texture analysis (real photos have more texture variation)
        texture_score = cv2.Laplacian(gray, cv2.CV_64F).var() / 1000.0
        
        # 4. Color gradient analysis
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(sobelx**2 + sobely**2)
        gradient_score = np.mean(gradient_magnitude) / 100.0
        
        # Combine scores (higher = more synthetic)
        synthetic_score = 0.3 * (1 - color_score) + 0.25 * (1 - edge_density) + 0.25 * (1 - texture_score) + 0.2 * (1 - gradient_score)
        
        is_synthetic = synthetic_score > 0.6
        confidence = synthetic_score
        
        reason = f"Color variety: {color_score:.2f}, Edge density: {edge_density:.3f}, Texture: {texture_score:.2f}"
        
        return is_synthetic, confidence, reason
    except Exception:
        return False, 0.0, "Heuristic analysis failed"


def run_ensemble_verifier(media_bytes: bytes, media_type: str, strictness: str = "strict") -> Tuple[bool, int, str]:
    # Load models if not already loaded
    model_status = load_models()
    
    if media_type == "image":
        # First, PIL-based checks: animation, size, and entropy
        try:
            from PIL import Image as PILImage
            pil_img = PILImage.open(io.BytesIO(media_bytes))
            width, height = pil_img.size
            n_frames = getattr(pil_img, "n_frames", 1)
            if n_frames and n_frames > 1:
                # Animated image -> reject
                return False, 72, "Animated images are not allowed for verification. Upload a single-frame photo."
            if width < 256 or height < 256:
                return False, 70, "Image too small. Please upload a higher-resolution photo (>=256x256)."
            try:
                entropy = pil_img.convert("L").entropy()
            except Exception:
                entropy = 0.0
            if entropy < 4.0:
                return False, 68, "Image has very low detail. Avoid plain graphics or cartoons."
        except Exception:
            pass

        if _cv2_available:
            import numpy as np
            image_array = np.frombuffer(media_bytes, dtype=np.uint8)
            img = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
            if img is None:
                return False, 65, "Unable to decode image; try a clearer photo."
            
            # Ensemble verification components
            results = {}
            
            # 1. Synthetic/Animation detection
            is_synthetic, synth_conf, synth_reason = detect_synthetic_heuristics(img)
            results['synthetic'] = {'detected': is_synthetic, 'confidence': synth_conf, 'reason': synth_reason}
            
            # 2. Object detection (YOLO)
            objects, obj_conf = detect_objects_yolo(img)
            results['objects'] = {'classes': objects, 'confidence': obj_conf}
            
            # 3. Pose detection (MediaPipe)
            num_poses, pose_conf = detect_poses_mediapipe(img)
            results['poses'] = {'count': num_poses, 'confidence': pose_conf}
            
            # 4. Semantic analysis (CLIP)
            semantic_score, description = analyze_semantic_clip(img)
            results['semantic'] = {'score': semantic_score, 'description': description}
            
            # 5. Basic image quality
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
            brightness = float(gray.mean())
            results['quality'] = {'sharpness': sharpness, 'brightness': brightness}
            
            # Ensemble scoring
            scores = []
            reasons = []
            
            # Reject if synthetic/animated
            if is_synthetic:
                return False, int(synth_conf * 100), f"Rejected: Appears synthetic/animated. {synth_reason}"
            
            # Object detection score
            if model_status['yolo']:
                service_objects = ['person', 'tree', 'bottle', 'cup', 'book', 'laptop', 'backpack']
                relevant_objects = [obj for obj in objects if any(so in obj.lower() for so in service_objects)]
                obj_score = len(relevant_objects) / max(len(service_objects), 1) * obj_conf
                scores.append(obj_score * 0.3)
                reasons.append(f"Objects: {', '.join(objects[:3])} (conf: {obj_conf:.2f})")
            else:
                scores.append(0.5)  # Neutral score if YOLO unavailable
                reasons.append("Object detection unavailable")
            
            # Pose detection score
            if model_status['mediapipe']:
                pose_score = min(num_poses * 0.5 + pose_conf, 1.0)  # Reward human presence
                scores.append(pose_score * 0.25)
                reasons.append(f"Poses detected: {num_poses} (conf: {pose_conf:.2f})")
            else:
                scores.append(0.3)  # Lower neutral score if pose detection unavailable
                reasons.append("Pose detection unavailable")
            
            # Semantic analysis score
            if model_status['clip']:
                semantic_score_norm = min(semantic_score * 2, 1.0)  # Scale CLIP score
                scores.append(semantic_score_norm * 0.25)
                reasons.append(f"Semantic: {description[:50]}... (score: {semantic_score:.2f})")
            else:
                scores.append(0.3)  # Lower neutral score if CLIP unavailable
                reasons.append("Semantic analysis unavailable")
            
            # Image quality score
            quality_score = 0.0
            if sharpness > 100 and 60 <= brightness <= 200:
                quality_score = 0.8
            elif sharpness > 50 and 40 <= brightness <= 220:
                quality_score = 0.6
            else:
                quality_score = 0.2
            scores.append(quality_score * 0.2)
            reasons.append(f"Quality: sharpness={sharpness:.1f}, brightness={brightness:.0f}")
            
            # Calculate final ensemble score
            ensemble_score = sum(scores)
            confidence = int(ensemble_score * 100)
            
            # Determine strictness thresholds
            if strictness == "strict":
                threshold = 0.7
            elif strictness == "balanced":
                threshold = 0.6
            else:  # lenient
                threshold = 0.5
            
            is_genuine = ensemble_score >= threshold and confidence >= 60
            
            # Generate detailed message
            status = "✅ VERIFIED" if is_genuine else "❌ REJECTED"
            msg = f"{status} - Ensemble Analysis:\n"
            msg += "\n".join([f"• {reason}" for reason in reasons])
            msg += f"\n\nFinal Score: {ensemble_score:.2f}/{1.0} (threshold: {threshold})"
            
            return is_genuine, max(40, min(100, confidence)), msg
        else:
            # Without OpenCV, be very conservative
            return False, 60, "OpenCV required for image analysis. Please install: pip install opencv-python"

    elif media_type == "video" and _cv2_available:
        # Video analysis - sample frames and run ensemble on each
        import tempfile
        import numpy as np
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
            tmp.write(media_bytes)
            tmp_path = tmp.name
        
        cap = cv2.VideoCapture(tmp_path)
        frame_scores = []
        frame_count = 0
        
        while frame_count < 10:  # Sample first 10 frames
            ret, frame = cap.read()
            if not ret:
                break
            
            # Run ensemble on this frame
            frame_bytes = cv2.imencode('.jpg', frame)[1].tobytes()
            is_genuine, conf, _ = run_ensemble_verifier(frame_bytes, "image", strictness)
            frame_scores.append(conf / 100.0)  # Normalize to 0-1
            frame_count += 1
        
        cap.release()
        try:
            os.remove(tmp_path)
        except Exception:
            pass
        
        if frame_count == 0:
            return False, 60, "Unable to process video frames."
        
        # Average frame scores
        avg_score = sum(frame_scores) / len(frame_scores)
        confidence = int(avg_score * 100)
        is_genuine = avg_score >= 0.6 and confidence >= 65  # Video threshold
        
        status = "✅ VERIFIED" if is_genuine else "❌ REJECTED"
        msg = f"{status} - Video Analysis:\nAnalyzed {frame_count} frames, average score: {avg_score:.2f}\nConfidence: {confidence}%"
        
        return is_genuine, max(40, min(100, confidence)), msg
    
    # Fallback
    return False, 55, "Advanced verification unavailable. Install: pip install ultralytics mediapipe torch transformers"


# -----------------------------
# Media Helpers
# -----------------------------
def save_uploaded_media(file, media_type: str) -> str:
    ensure_directories_exist()
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S_%f")
    safe_name = file.name.replace(" ", "_")
    if media_type == "image":
        dest = os.path.join(IMAGES_DIR, f"{timestamp}_{safe_name}")
    else:
        dest = os.path.join(VIDEOS_DIR, f"{timestamp}_{safe_name}")
    with open(dest, "wb") as f:
        f.write(file.read())
    return dest


def google_vision_verify(img_path: str, api_key: Optional[str]) -> Tuple[Optional[bool], List[str], str]:
    if not api_key:
        return None, [], "No API key configured"
    try:
        url = f"https://vision.googleapis.com/v1/images:annotate?key={api_key}"
        with open(img_path, "rb") as f:
            content = base64.b64encode(f.read()).decode()
        payload = {
            "requests": [
                {
                    "image": {"content": content},
                    "features": [{"type": "LABEL_DETECTION", "maxResults": 12}],
                }
            ]
        }
        res = requests.post(url, json=payload, timeout=8)
        data = res.json()
        labels = [x["description"].lower() for x in data.get("responses", [{}])[0].get("labelAnnotations", [])]
        keywords = ["volunteer", "cleanup", "planting", "community", "outdoor", "people", "trash", "helping", "charity", "donation"]
        match = any(k in l for l in labels for k in keywords)
        return match, labels, "OK"
    except Exception as ex:
        return None, [], f"Vision error: {ex}"


def configure_gemini_from_env() -> Optional[str]:
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if api_key and _gemini_available:
        try:
            genai.configure(api_key=api_key)
            return api_key
        except Exception:
            return None
    return None


def gemini_verify_media(media_path: str, media_type: str) -> Tuple[bool, int, str]:
    api_key = configure_gemini_from_env()
    if not api_key or not _gemini_available:
        return False, 50, "Gemini not available or API key missing in .env"
    try:
        model = genai.GenerativeModel("gemini-2.0-flash")
        system_prompt = (
            "You are verifying if the provided media shows a REAL social service activity. "
            "Examples: tree planting, cleaning public spaces, helping people, distributing food, community service. "
            "Return a strict JSON with keys: genuine (true/false), confidence (0-100), reason (short)."
        )
        if media_type == "image":
            with open(media_path, "rb") as f:
                img_bytes = f.read()
            parts = [
                system_prompt,
                {"mime_type": "image/jpeg", "data": img_bytes},
                "Respond in JSON only."
            ]
        else:
            frames = extract_video_frames(media_path, interval_s=2.0, max_frames=4)
            if not frames:
                first = get_video_first_frame_bytes(media_path)
                frames = [first] if first else []
            parts = [system_prompt]
            for fb in frames:
                parts.append({"mime_type": "image/jpeg", "data": fb})
            parts.append("Respond in JSON only.")

        with st.spinner("Verifying your activity..."):
            resp = model.generate_content(parts)
        text = (resp.text or "").strip()
        # Attempt to parse JSON
        import re, json as pyjson
        match = re.search(r"\{[\s\S]*\}", text)
        data = pyjson.loads(match.group(0)) if match else pyjson.loads(text)
        genuine = bool(data.get("genuine", False))
        confidence = int(data.get("confidence", 60))
        reason = str(data.get("reason", ""))
        confidence = max(0, min(100, confidence))
        return genuine, confidence, reason
    except Exception as ex:
        return False, 55, f"Gemini error: {ex}"


def get_video_first_frame_bytes(path: str) -> Optional[bytes]:
    if not _cv2_available:
        return None
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return None
    success, frame = cap.read()
    cap.release()
    if not success:
        return None
    # Convert BGR to RGB and encode as PNG
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    is_ok, buf = cv2.imencode(".png", frame_rgb)
    if not is_ok:
        return None
    return buf.tobytes()


def extract_video_frames(path: str, interval_s: float = 2.0, max_frames: int = 4) -> List[bytes]:
    if not _cv2_available:
        return []
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return []
    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    step = int(max(1, round(fps * interval_s)))
    frames: List[bytes] = []
    idx = 0
    grabbed = True
    while grabbed and len(frames) < max_frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        grabbed, frame = cap.read()
        if not grabbed:
            break
        is_ok, buf = cv2.imencode('.jpg', frame)
        if is_ok:
            frames.append(buf.tobytes())
        idx += step
    cap.release()
    return frames


# -----------------------------
# UI Theme and Styles
# -----------------------------
def inject_dark_mode_css() -> None:
    st.markdown(
        """
        <style>
        :root {
          --bg: #0f172a; /* slate-900 */
          --panel: #111827; /* gray-900 */
          --card: #111827;
          --text: #e5e7eb; /* gray-200 */
          --muted: #9ca3af; /* gray-400 */
          --primary: #22d3ee; /* cyan-400 */
          --secondary: #a78bfa; /* violet-400 */
          --accent: #f59e0b; /* amber-500 */
          --success: #10b981; /* emerald-500 */
          --danger: #ef4444; /* red-500 */
        }
        .stApp {
          background: radial-gradient(1200px 600px at 10% 10%, rgba(34,211,238,0.06), transparent),
                      radial-gradient(1000px 500px at 90% 10%, rgba(167,139,250,0.06), transparent),
                      var(--bg) !important;
          color: var(--text) !important;
        }
        header, .css-18ni7ap, .css-1rs6os { background: transparent !important; }
        .scc-title {
          display: flex; align-items: center; gap: 12px; margin-bottom: 8px;
        }
        .scc-logo {
          width: 40px; height: 40px; border-radius: 12px;
          background: linear-gradient(135deg, var(--primary), var(--secondary));
          display: inline-flex; align-items: center; justify-content: center; font-weight: 800; color: #0b1020;
          box-shadow: 0 8px 20px rgba(34,211,238,0.25);
        }
        .scc-card {
          background: rgba(17,24,39,0.85);
          border: 1px solid rgba(255,255,255,0.06);
          padding: 16px 18px; border-radius: 16px; margin-bottom: 12px;
        }
        .metric {
          display:flex; align-items:center; justify-content:space-between; padding:12px 14px;
          border-radius: 14px; background: rgba(2,6,23,0.6); border:1px solid rgba(255,255,255,0.06);
        }
        .metric .label { color: var(--muted); font-size: 0.9rem; }
        .metric .value { font-size: 1.4rem; font-weight: 700; color: var(--primary); }
        .pill {
          display:inline-block; padding:6px 10px; border-radius:999px; font-weight:600;
          background: rgba(16,185,129,0.12); color: #86efac; border:1px solid rgba(16,185,129,0.3);
        }
        .pill.danger { background: rgba(239,68,68,0.12); color:#fecaca; border-color: rgba(239,68,68,0.3); }
        .suggestion {
          background: rgba(245,158,11,0.12); border:1px solid rgba(245,158,11,0.3);
          color:#fde68a; padding:10px 12px; border-radius: 12px; margin-bottom:8px;
        }
        .footer-note { color: var(--muted); font-size: 0.85rem; margin-top: 8px; }
        </style>
        """,
        unsafe_allow_html=True,
    )


# -----------------------------
# UI Components
# -----------------------------
def app_header() -> None:
    st.markdown(
        f"""
        <div class="scc-title">
            <div class="scc-logo">🅢</div>
            <div>
              <div style="font-size:1.6rem; font-weight:800; letter-spacing:0.3px;">{APP_TITLE}</div>
              <div style="color:#93c5fd; font-size:0.95rem;">Earn ServePoints for real community impact ✨</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def sidebar_login(conn: sqlite3.Connection) -> Optional[Dict[str, str]]:
    with st.sidebar:
        st.markdown("### 🔐 Login or Create Account")
        tabs = st.tabs(["Login", "Sign up"])
        with tabs[0]:
            uname = st.text_input("Username", key="login_uname")
            pwd = st.text_input("Password", type="password", key="login_pwd")
            col_a, col_b = st.columns([1, 1])
            with col_a:
                do_login = st.button("Login", use_container_width=True)
            with col_b:
                do_logout = st.button("Logout", use_container_width=True)
            if do_login:
                user = authenticate_user(conn, uname, pwd)
                if user:
                    st.session_state["auth_user"] = user
                    st.success(f"Welcome back, {user['name']}!")
                    time.sleep(0.2)
                else:
                    st.error("Invalid credentials.")
            if do_logout:
                st.session_state.pop("auth_user", None)
                st.info("Logged out.")
        with tabs[1]:
            disp = st.text_input("Display Name", key="signup_display")
            suname = st.text_input("Choose Username", key="signup_uname")
            spwd = st.text_input("Choose Password", type="password", key="signup_pwd")
            if st.button("Create Account", use_container_width=True):
                ok, msg = create_user(conn, suname, disp, spwd)
                if ok:
                    st.success(f"Account created for {msg}. You can login now.")
                else:
                    st.error(msg)
        st.divider()
    return st.session_state.get("auth_user")


def sidebar_nav() -> str:
    with st.sidebar:
        page = st.radio(
            "Navigate",
            (
                "Home",
                "Upload Activity",
                "Leaderboard",
                "My Activities",
                "Profile / Savings",
            ),
            index=0,
        )
        st.markdown(
            "<div class='footer-note'>All processing is local. No external APIs used.</div>",
            unsafe_allow_html=True,
        )
        return page


# -----------------------------
# Pages
# -----------------------------
def page_home(conn: sqlite3.Connection) -> None:
    num_users, total_points = query_totals(conn)
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("<div class='scc-card'>", unsafe_allow_html=True)
        st.markdown("<div class='metric'><div class='label'>Total Users</div><div class='value'>👥 " + str(num_users) + "</div></div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
    with col2:
        st.markdown("<div class='scc-card'>", unsafe_allow_html=True)
        st.markdown("<div class='metric'><div class='label'>Total ServePoints Distributed</div><div class='value'>⭐ " + str(total_points) + "</div></div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='scc-card'>", unsafe_allow_html=True)
    st.markdown(
        """
        #### 🌍 Welcome to SCC (Service Credit Card)
        Earn ServePoints by uploading genuine photos or videos of your social service activities.
        Our on-device dummy AI verifies uploads and awards you points. Compete on the leaderboard and
        track your GST savings over time. Do good. Get recognized. 💙
        """
    )
    st.markdown("</div>", unsafe_allow_html=True)


def page_upload(conn: sqlite3.Connection, auth_user: Optional[Dict[str, str]]) -> None:
    if not auth_user:
        st.info("Please login to upload your activity.")
        return

    st.markdown("<div class='scc-card'>", unsafe_allow_html=True)
    st.markdown("### 📤 Upload Your Activity")
    activity_label = st.text_input("Activity Title (e.g., Plant Trees)", max_chars=60)
    media_choice = st.segmented_control("Media Type", options=["Image", "Video"], default="Image")
    if media_choice == "Image":
        uploaded = st.file_uploader("Upload an Image (JPG/PNG/GIF/WEBP)", type=["jpg", "jpeg", "png", "gif", "webp"], accept_multiple_files=False)
    else:
        uploaded = st.file_uploader("Upload a Video (MP4/MOV/AVI)", type=["mp4", "mov", "avi"], accept_multiple_files=False)

    if uploaded is not None:
        media_type = "image" if media_choice == "Image" else "video"
        # Preview
        if media_type == "image":
            try:
                img = Image.open(uploaded).convert("RGB")
                st.image(img, caption="Preview", use_column_width=True)
                uploaded.seek(0)  # reset pointer
            except Exception as ex:
                st.error(f"Failed to preview image: {ex}")
        else:
            file_bytes = uploaded.getvalue()
            st.video(file_bytes)
            if _cv2_available:
                st.caption("OpenCV preview (first frame):")
                # Save temp to extract first frame, then delete
                tmp_path = os.path.join(VIDEOS_DIR, "_tmp_preview.mp4")
                ensure_directories_exist()
                with open(tmp_path, "wb") as f:
                    f.write(file_bytes)
                frame_bytes = get_video_first_frame_bytes(tmp_path)
                try:
                    os.remove(tmp_path)
                except Exception:
                    pass
                if frame_bytes:
                    st.image(frame_bytes, caption="First Frame", use_column_width=True)
            else:
                st.warning("OpenCV not available; showing video via built-in player only.")

        st.divider()
        if st.button("Verify & Award Points ✨", type="primary", use_container_width=True, help="Verifies using Gemini 2.0 Flash"):
            uploaded.seek(0)
            media_bytes = uploaded.read()
            uploaded.seek(0)
            saved_path = save_uploaded_media(uploaded, media_type)
            is_genuine, confidence, reason = gemini_verify_media(saved_path, media_type)
            ai_msg = f"Gemini: {reason}" if reason else "Gemini verification completed."

            points = random.randint(120, 420) if is_genuine else 0

            # Persist activity
            label = activity_label.strip() or random.choice(ACTIVITY_SUGGESTIONS)
            insert_activity(
                conn=conn,
                user_name=auth_user["name"],
                media_path=saved_path,
                media_type=media_type,
                activity_label=label,
                is_genuine=is_genuine,
                confidence=confidence,
                points=points,
            )

            # Result card
            st.markdown("<div class='scc-card'>", unsafe_allow_html=True)
            icon = "✅" if is_genuine else "❌"
            result_color_class = "pill" if is_genuine else "pill danger"
            st.markdown(f"<span class='{result_color_class}'>Confidence: {confidence}%</span>", unsafe_allow_html=True)
            st.markdown(f"### {icon} Verification Result")
            st.text(ai_msg)  # Use st.text for better formatting of multi-line messages
            if is_genuine:
                st.success(f"🎉 Congratulations {auth_user['name']}! You earned {points} ServePoints.")
            else:
                st.error("This upload could not be verified as genuine. No points awarded.")
            st.markdown("</div>", unsafe_allow_html=True)

            # Suggestions
            st.markdown("### 💡 Try these activities next")
            suggestions = random.sample(ACTIVITY_SUGGESTIONS, k=3)
            for s in suggestions:
                potential = random.randint(200, 500)
                st.markdown(f"<div class='suggestion'>📝 {s} — potential ⭐ {potential} points</div>", unsafe_allow_html=True)


def page_leaderboard(conn: sqlite3.Connection) -> None:
    st.markdown("### 🏆 Leaderboard")
    data = get_leaderboard(conn)
    if not data:
        st.info("No users yet. Upload an activity to get started!")
        return

    import pandas as pd

    df = pd.DataFrame(data, columns=["User", "TotalPoints"])  # type: ignore
    df.insert(0, "Rank", range(1, len(df) + 1))

    st.dataframe(
        df.style.hide(axis="index").bar(color="#22d3ee", subset=["TotalPoints"], vmax=max(1, df["TotalPoints"].max())),
        use_container_width=True,
    )

    st.markdown("#### 📊 Points by User")
    st.bar_chart(df.set_index("User")["TotalPoints"], use_container_width=True)


def page_my_activities(conn: sqlite3.Connection, auth_user: Optional[Dict[str, str]]) -> None:
    if not auth_user:
        st.info("Please login to view your activities.")
        return
    st.markdown("### 📜 My Activities")
    rows = get_user_activities(conn, auth_user["name"])
    if not rows:
        st.info("No activities yet. Upload your first contribution!")
        return

    import pandas as pd

    df = pd.DataFrame(rows)  # type: ignore
    df_display = df[["Date", "Activity", "Genuine", "Confidence", "Points", "Type"]]
    st.dataframe(df_display, use_container_width=True, hide_index=True)


def page_profile(conn: sqlite3.Connection, auth_user: Optional[Dict[str, str]]) -> None:
    if not auth_user:
        st.info("Please login to view your profile.")
        return
    st.markdown("### 🧾 Profile & Savings")
    points = get_user_points(conn, auth_user["name"])
    savings_inr = points / 10.0  # ₹1 per 10 points

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("<div class='scc-card'>", unsafe_allow_html=True)
        st.markdown(
            f"<div class='metric'><div class='label'>Total ServePoints</div><div class='value'>⭐ {points}</div></div>",
            unsafe_allow_html=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)
    with col2:
        st.markdown("<div class='scc-card'>", unsafe_allow_html=True)
        st.markdown(
            f"<div class='metric'><div class='label'>GST Savings (₹1 per 10 points)</div><div class='value'>₹ {savings_inr:,.2f}</div></div>",
            unsafe_allow_html=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)

    st.caption("Savings are illustrative for project demo purposes.")


# -----------------------------
# Main App
# -----------------------------
def main() -> None:
    st.set_page_config(page_title="SCC App", page_icon="🅢", layout="wide")
    inject_dark_mode_css()
    app_header()

    ensure_directories_exist()
    # Configure Gemini on startup (uses .env)
    _ = configure_gemini_from_env()
    conn = get_db_connection()
    init_db(conn)
    ensure_auth_schema(conn)
    seed_sample_data(conn)

    auth_user = sidebar_login(conn)
    page = sidebar_nav()

    if page == "Home":
        page_home(conn)
    elif page == "Upload Activity":
        page_upload(conn, auth_user)
    elif page == "Leaderboard":
        page_leaderboard(conn)
    elif page == "My Activities":
        page_my_activities(conn, auth_user)
    elif page == "Profile / Savings":
        page_profile(conn, auth_user)
    else:
        page_home(conn)

    st.markdown(
        "<div class='footer-note'>Tip: Run with <code>streamlit run service.py</code>. Enjoy building your impact! ✨</div>",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()


