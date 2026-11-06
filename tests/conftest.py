import sys
from pathlib import Path

# 让 tests/ 下的文件能直接 import model / dataset 包
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
