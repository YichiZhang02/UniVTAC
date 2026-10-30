"""Train/export the fixed Sparsh_IJEPA architecture."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from encoder.train_external import main
if __name__ == '__main__':
    main('Sparsh_IJEPA', 'tactile')
