"""ZJU-MoCap loader (raw EasyMocap / Neural-Body layout).

Layout expected under `cfg.data_root` (a subject folder, e.g. CoreView_377):
    annots.npy                 # {'cams': {K,R,T,D}, 'ims': [frame]['ims'][cam]}
    images/<cam>/<frame>.jpg   # (paths come from annots['ims'])
    mask/<cam>/<frame>.png  or  mask_cihp/<cam>/<frame>.png
    new_params/<frame>.npy  or  params/<frame>.npy   # poses(72),shapes(10),Rh(3),Th(3)

One sample = (image, mask, Camera, smpl{poses,shapes,Rh,Th}, frame_id, cam_id).
Conventions handled: R is world->cam, T may be in mm (cfg.t_in_mm), principal
point from K, intrinsics scaled with the resize factor (camera.make_camera).
"""
from __future__ import annotations
import os
import numpy as np
import torch
from PIL import Image
from camera import make_camera


def _load_image(path, scale):
    img = Image.open(path).convert("RGB")
    if scale != 1.0:
        img = img.resize((round(img.width * scale), round(img.height * scale)),
                         Image.BILINEAR)
    return torch.from_numpy(np.asarray(img, np.float32) / 255.0).permute(2, 0, 1)


def _load_mask(path, scale):
    m = Image.open(path).convert("L")
    if scale != 1.0:
        m = m.resize((round(m.width * scale), round(m.height * scale)), Image.NEAREST)
    # ZJU masks are stored as 0/1 (mask/) or small label ids (mask_cihp/), not 0/255,
    # so treat any positive pixel as foreground (a >127 threshold blanks them out).
    return torch.from_numpy((np.asarray(m, np.float32) > 0).astype(np.float32))


def _find(root, rel, exts):
    for e in exts:
        p = os.path.join(root, os.path.splitext(rel)[0] + e)
        if os.path.exists(p):
            return p
    return None


class ZJUDataset:
    def __init__(self, cfg, split="train"):
        self.cfg = cfg
        self.root = cfg.data_root
        annots = np.load(os.path.join(self.root, "annots.npy"), allow_pickle=True).item()
        self.cams = annots["cams"]
        ims = annots["ims"]

        n_frames = min(cfg.train_frames, len(ims))
        n_hold = max(1, int(len(ims) * cfg.demo_frames_holdout))
        if split == "train":
            frames = list(range(n_frames))
        elif split == "all":          # whole captured take (longer demo clip)
            frames = list(range(len(ims)))
        else:  # demo / holdout = last frames
            frames = list(range(len(ims) - n_hold, len(ims)))

        cam_ids = list(cfg.train_views) if cfg.train_views else \
            list(range(len(self.cams["K"])))

        self.samples = []
        for fi in frames:
            for ci in cam_ids:
                self.samples.append((fi, ci))
        self.ims = ims
        self.cam_ids = cam_ids
        self.scale = cfg.image_scale

    def __len__(self):
        return len(self.samples)

    def rest_betas(self):
        """Canonical shape = first training frame's SMPL betas."""
        return self._load_smpl(0)["shapes"]

    def _load_smpl(self, frame_idx):
        for sub in ("new_params", "params"):
            p = os.path.join(self.root, sub, f"{frame_idx}.npy")
            if os.path.exists(p):
                d = np.load(p, allow_pickle=True).item()
                return {
                    "poses": torch.tensor(np.asarray(d["poses"], np.float32)).reshape(-1),
                    "shapes": torch.tensor(np.asarray(
                        d.get("shapes", d.get("betas")), np.float32)).reshape(-1),
                    "Rh": torch.tensor(np.asarray(d["Rh"], np.float32)).reshape(3),
                    "Th": torch.tensor(np.asarray(d["Th"], np.float32)).reshape(3),
                }
        raise FileNotFoundError(f"No SMPL params for frame {frame_idx} in {self.root}")

    def _camera(self, cam_id):
        K = np.asarray(self.cams["K"][cam_id], np.float32)
        R = np.asarray(self.cams["R"][cam_id], np.float32)
        T = np.asarray(self.cams["T"][cam_id], np.float32)
        # full-res size from the first image of this camera
        rel = self.ims[0]["ims"][cam_id]
        with Image.open(os.path.join(self.root, rel)) as im:
            W, H = im.size
        return make_camera(K, R, T, scale=self.scale, t_in_mm=self.cfg.t_in_mm,
                           W=W, H=H, device=self.cfg.device)

    def __getitem__(self, i):
        frame_idx, cam_id = self.samples[i]
        rel = self.ims[frame_idx]["ims"][cam_id]
        img_path = os.path.join(self.root, rel)
        # masks live under mask/ or mask_cihp/ mirroring the image path
        mrel = rel.replace("images/", "") if rel.startswith("images/") else rel
        mask_path = (_find(os.path.join(self.root, "mask"), mrel, [".png", ".jpg"])
                     or _find(os.path.join(self.root, "mask_cihp"), mrel, [".png", ".jpg"]))

        image = _load_image(img_path, self.scale).to(self.cfg.device)
        mask = (_load_mask(mask_path, self.scale).to(self.cfg.device)
                if mask_path else (image.sum(0) > 0).float())
        image = image * mask                                   # blacken background

        return {
            "image": image,                                    # (3,H,W) in [0,1]
            "mask": mask,                                      # (H,W) in {0,1}
            "camera": self._camera(cam_id),
            "smpl": self._load_smpl(frame_idx),
            "frame_id": frame_idx,
            "cam_id": cam_id,
        }
