from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "raw"
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS_HEAD = 3      # ヘッド学習
EPOCHS_FINETUNE = 12 # ファインチューニング
SEED = 42

MODEL_DIR = PROJECT_ROOT / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)
MODEL_PATH = MODEL_DIR / "gymbar_classifier.keras"
LABELS_PATH = MODEL_DIR / "labels.json"