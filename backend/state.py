from __future__ import annotations

from typing import Optional

import torch

from ml.model import SiameseDamageNet

model: Optional[SiameseDamageNet] = None
device: Optional[torch.device] = None
