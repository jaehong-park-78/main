"""
FWR ASI Growth Engine v3.1
W 설계 수정: 랜덤 대신 모델 내부에서 실제 구조 정보 추출

변경 핵심:
- W = 모델 activation 통계 기반 (레이어별 주파수 응답)
- F = 예측 확신도 (기존 유지)
- R = F-W 실제 상관관계 반영
- WExtractor 모듈 추가
"""

import torch
import torch.nn as nn
import torch.nn.functional as F_pt
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
from typing import List, Dict, Tuple, Optional
import math
import time


# ============================================
