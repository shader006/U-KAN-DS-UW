import os
import sys
import random
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import albumentations as A
import cv2
import numpy as np
import pytorch_lightning as pl
import torch
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader, Dataset
from pytorch_lightning.callbacks import ModelCheckpoint

from loss import bce_dice_loss, calculate_metrics, LearnableDSUncertaintyLoss

# Make U-KAN segmentation model importable
ROOT_DIR = os.path.dirname(__file__)
UKAN_SEG_DIR = os.path.join(ROOT_DIR, "U-KAN", "Seg_UKAN")
if UKAN_SEG_DIR not in sys.path:
    sys.path.append(UKAN_SEG_DIR)
from archs import UKAN  # noqa: E402


def set_seed(seed: int = 42) -> None:
    # Needed for deterministic CuBLAS operations when CUDA >= 10.2
    if "CUBLAS_WORKSPACE_CONFIG" not in os.environ:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)


class KvasirDataset(Dataset):
    def __init__(self, root_dir: str, phase: str = "train", transform: Optional[A.BasicTransform] = None):
        self.transform = transform
        self.images_dir = os.path.join(root_dir, phase, "images")
        self.masks_dir = os.path.join(root_dir, phase, "masks")
        self.image_ids = sorted(os.listdir(self.images_dir))

    def __len__(self) -> int:
        return len(self.image_ids)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        img_name = self.image_ids[idx]
        img_path = os.path.join(self.images_dir, img_name)
        mask_path = os.path.join(self.masks_dir, img_name)

        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        _, mask = cv2.threshold(mask, 127, 1, cv2.THRESH_BINARY)
        mask = mask.astype("float32")

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented["image"]
            mask = augmented["mask"]

        return image, mask


def build_transforms(img_size: int):
    train_transform = A.Compose(
        [
            A.Resize(img_size, img_size),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=20, p=0.7),
            A.ElasticTransform(alpha=120, sigma=12, p=0.3),
            A.RandomBrightnessContrast(p=0.3),
            A.HueSaturationValue(p=0.2),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ]
    )

    eval_transform = A.Compose(
        [
            A.Resize(img_size, img_size),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ]
    )

    return train_transform, eval_transform


@dataclass
class Config:
    root_dir: str = r"d:\code\DS+UW+Proof"
    img_size: int = 352
    batch_size: int = 8
    val_batch_size: int = 8
    num_workers: int = 4
    lr: float = 1e-4
    max_epochs: int = 200
    deep_supervision: bool = True
    use_uncertainty: bool = True
    aux_near_weight: float = 0.5
    aux_far_weight: float = 0.25
    seed: int = 42


class LitDataModule(pl.LightningDataModule):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.train_transform, self.eval_transform = build_transforms(cfg.img_size)

    def setup(self, stage: Optional[str] = None):
        self.train_ds = KvasirDataset(self.cfg.root_dir, phase="train", transform=self.train_transform)
        self.val_ds = KvasirDataset(self.cfg.root_dir, phase="val", transform=self.eval_transform)

    def train_dataloader(self):
        g = torch.Generator().manual_seed(self.cfg.seed)
        return DataLoader(
            self.train_ds,
            batch_size=self.cfg.batch_size,
            shuffle=True,
            num_workers=self.cfg.num_workers,
            pin_memory=True,
            generator=g,
            persistent_workers=self.cfg.num_workers > 0,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds,
            batch_size=self.cfg.val_batch_size,
            shuffle=False,
            num_workers=self.cfg.num_workers,
            pin_memory=True,
            persistent_workers=self.cfg.num_workers > 0,
        )


def _split_heads(outputs):
    """
    Normalize deep-supervision outputs to (main, aux_near, aux_far).
    Expected order: [main, aux_near, aux_far]. Extra heads are ignored.
    """
    if not isinstance(outputs, (list, tuple)):
        return outputs, None, None
    main_out = outputs[0] if len(outputs) > 0 else outputs
    aux_near = outputs[1] if len(outputs) > 1 else None
    aux_far = outputs[2] if len(outputs) > 2 else None
    return main_out, aux_near, aux_far


class LitUNet(pl.LightningModule):
    def __init__(self, cfg: Config):
        super().__init__()
        self.save_hyperparameters()
        self.cfg = cfg
        self.n_heads = 3 if cfg.deep_supervision else 1
        # Swap in UKAN segmentation model (returns [main, aux_near, aux_far] when training with DS)
        self.model = UKAN(
            num_classes=1,
            input_channels=3,
            deep_supervision=cfg.deep_supervision,
        )

        if cfg.use_uncertainty and cfg.deep_supervision:
            self.criterion = LearnableDSUncertaintyLoss(
                base_loss=bce_dice_loss,
                n_heads=self.n_heads,
                main_index=0,
                loss_types=tuple("cls" for _ in range(self.n_heads)),
                interp_mode="bilinear",
            )
        else:
            self.criterion = bce_dice_loss

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        images, masks = batch
        masks = masks.unsqueeze(1)
        outputs = self(images)
        main_out, aux_near, aux_far = _split_heads(outputs)
        heads = [h for h in (main_out, aux_near, aux_far) if h is not None]

        if self.cfg.use_uncertainty and self.cfg.deep_supervision:
            # Ensure we pass the expected number of heads to the uncertainty loss
            if len(heads) < self.n_heads:
                heads = heads + [main_out] * (self.n_heads - len(heads))
            loss = self.criterion(heads[: self.n_heads], masks)
        else:
            loss = bce_dice_loss(main_out, masks)
            if aux_near is not None:
                loss = loss + self.cfg.aux_near_weight * bce_dice_loss(aux_near, masks)
            if aux_far is not None:
                loss = loss + self.cfg.aux_far_weight * bce_dice_loss(aux_far, masks)

        self.log("train/loss", loss, prog_bar=True, on_step=False, on_epoch=True, batch_size=images.size(0))
        return loss

    def validation_step(self, batch, batch_idx):
        images, masks = batch
        masks = masks.unsqueeze(1)
        outputs = self(images)
        main_out, _, _ = _split_heads(outputs)
        loss = bce_dice_loss(main_out, masks)

        d, iou, p, r = calculate_metrics(main_out, masks)
        self.log("val/loss", loss, prog_bar=True, on_step=False, on_epoch=True, batch_size=images.size(0))
        self.log("val/dice", d, prog_bar=True, on_step=False, on_epoch=True, batch_size=images.size(0))
        self.log("val/iou", iou, on_step=False, on_epoch=True, batch_size=images.size(0))
        self.log("val/precision", p, on_step=False, on_epoch=True, batch_size=images.size(0))
        self.log("val/recall", r, on_step=False, on_epoch=True, batch_size=images.size(0))

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.cfg.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", patience=5, factor=0.5)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val/dice",
                "interval": "epoch",
                "frequency": 1,
            },
        }


def main():
    cfg = Config()
    set_seed(cfg.seed)

    data_module = LitDataModule(cfg)
    model = LitUNet(cfg)

    checkpoint_cb = ModelCheckpoint(
        monitor="val/dice",
        mode="max",
        save_top_k=1,
        filename="unet-{epoch:02d}-{val_dice:.4f}",
        save_last=True,
    )

    trainer = pl.Trainer(
        max_epochs=cfg.max_epochs,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        precision="16-mixed" if torch.cuda.is_available() else 32,
        log_every_n_steps=10,
        callbacks=[checkpoint_cb],
    )
    trainer.fit(model, datamodule=data_module)


if __name__ == "__main__":
    main()
