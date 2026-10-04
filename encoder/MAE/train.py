"""Train the MAE tactile encoder (ViT-S or ViT-B)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from train_common import main
if __name__ == '__main__':
    main('MAE')
