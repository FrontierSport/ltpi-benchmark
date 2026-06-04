import torch
import torch.nn as nn
from torchvision import models
import warnings
from typing import Optional, Tuple

warnings.filterwarnings("ignore")


class MLPProjector(nn.Module):
    def __init__(self, embedding_dim: int, num_classes: int = 3) -> None:
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim // 2),
            nn.Dropout(p=0.1),
            nn.ReLU(inplace=True),
            nn.Linear(embedding_dim // 2, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


class MobileNetEmbedding(nn.Module):
    def __init__(
        self,
        embedding_size: int = 128,
        num_classes: int = 3,
        pretrained: bool = True,
        freeze_layers: bool = True,
    ) -> None:
        super().__init__()
        self.backbone = models.mobilenet_v3_large(pretrained=pretrained)
        num_ftrs = self.backbone.classifier[3].in_features
        self.backbone.classifier[3] = nn.Linear(num_ftrs, embedding_size)

        if freeze_layers:
            for param in self.backbone.features.parameters():
                param.requires_grad = False

        self.linear_head = MLPProjector(embedding_size, num_classes)

    def forward(
        self,
        x: Optional[torch.Tensor] = None,
        user_embedding: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if x is not None:
            embedding = self.backbone(x)
        elif user_embedding is not None:
            embedding = user_embedding
        else:
            raise ValueError("Model input cannot be None — provide x or user_embedding.")

        logits = self.linear_head(embedding)
        return logits, embedding