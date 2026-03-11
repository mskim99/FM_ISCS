from pathlib import Path
from typing import Any, Dict, Optional

import torch
import torch.nn.functional as F

from tqdm import tqdm

import algorithms.utils as autils
from models import utils as mutils
from utils.result import save_nii_image
from torchvision.utils import save_image

def _save_debug_tensor(x, path: Path):
    """
    x: [B, C, H, W] tensor
    첫 번째 샘플만 저장
    """
    y = x[:1].detach().float().cpu()

    # 시각화용 정규화
    y = y - y.min()
    if y.max() > 0:
        y = y / y.max()

    path.parent.mkdir(parents=True, exist_ok=True)
    save_image(y, str(path))
    print(f"[DEBUG] saved: {path}", flush=True)

class DDS:
    def __init__(self, sde: Any, model: Any, config: Any, measure_model: Any, eps: float = 1e-10, factor: int = 4, plane="axial"):
        self.sde = sde
        self.model = model
        self.config = config
        self.measure_model = measure_model
        self.eps = eps
        self.plane = plane

        # Precompute score function
        self.score_fn = mutils.get_score_fn(
            self.sde, self.model, train=False, continuous=self.config.training.continuous
        )
        self.factor = factor
        self.M = autils.get_M(factor)
        self.invM = torch.inverse(self.M)

    def _A(self, x: torch.Tensor) -> torch.Tensor:
        # return self.measure_model.A(x)
        return self._A(x)

    def _AT(self, y: torch.Tensor) -> torch.Tensor:
        # ATy = self.measure_model.A_dagger(y)
        ATy = self._A_dagger(y)
        # ATy = self.measure_model.couple(self.measure_model.decouple(ATy))
        ATy = self._couple_decouple(ATy)
        return ATy

    def A_cg(self, x: torch.Tensor) -> torch.Tensor:
        return self._AT(self._A(x))

    def A_cg_tik(self, x: torch.Tensor, rho: float) -> torch.Tensor:
        return self._AT(self._A(x)) + rho * x

    def _vol_to_score(self, x):
        """
        canonical volume [Z,1,Y,X] -> score model batch [B,1,H,W]
        """
        if self.plane == "axial":
            return x  # [Z,1,Y,X]
        if self.plane == "coronal":
            return x.permute(2, 1, 0, 3)  # [Y,1,Z,X]
        if self.plane == "sagittal":
            return x.permute(3, 1, 0, 2)  # [X,1,Z,Y]
        raise ValueError(self.plane)

    def _score_to_vol(self, x):
        """
        score model batch [B,1,H,W] -> canonical volume [Z,1,Y,X]
        """
        if self.plane == "axial":
            return x
        if self.plane == "coronal":
            return x.permute(2, 1, 0, 3)  # [Z,1,Y,X]
        if self.plane == "sagittal":
            return x.permute(2, 1, 3, 0)  # [Z,1,Y,X]
        raise ValueError(self.plane)

    def _vol_to_phys(self, x):
        """
        canonical volume [Z,1,Y,X] -> physics batch [X,1,Y,Z]
        마지막 축이 항상 z
        """
        return x.permute(3, 1, 2, 0)

    def _phys_to_vol(self, x):
        """
        physics batch [X,1,Y,Z] -> canonical volume [Z,1,Y,X]
        """
        return x.permute(3, 1, 2, 0)

    def _A(self, x_vol):
        """
        canonical volume -> measurement domain
        """
        x_phys = self._vol_to_phys(x_vol)
        return self.measure_model.A(x_phys)

    def _A_dagger(self, y):
        """
        measurement domain -> canonical volume
        """
        x_phys = self.measure_model.A_dagger(y)
        return self._phys_to_vol(x_phys)

    def _couple_decouple(self, x_vol):
        """
        canonical volume -> canonical volume
        """
        x_phys = self._vol_to_phys(x_vol)
        x_phys = self.measure_model.couple(self.measure_model.decouple(x_phys))
        return self._phys_to_vol(x_phys)

    def _get_mask(self, x_vol, channel):
        """
        필요할 경우 mask도 canonical 기준으로 반환
        """
        x_phys = self._vol_to_phys(x_vol)
        mask_phys = self.measure_model.get_mask(x_phys, channel)
        return self._phys_to_vol(mask_phys)

    def tweedie_denoising_fn(self, x_score: torch.Tensor, t: torch.Tensor) -> tuple:

        '''
        vec_t = torch.ones(x.shape[0], device=x.device) * t
        with torch.no_grad():
            score_t = self.score_fn(x, vec_t)
        sigma_t = self.sde.sigma_min * (self.sde.sigma_max / self.sde.sigma_min) ** t
        x_0_hat = x + (sigma_t**2) * score_t
        '''
        vec_t = torch.ones(x_score.shape[0], device=x_score.device) * t
        score_score = self.score_fn(x_score, vec_t)

        sigma_t = self.sde.marginal_prob(
            torch.zeros((1, 1, x_score.shape[-2], x_score.shape[-1]), device=x_score.device),
            torch.tensor([t], device=x_score.device),
        )[1][0]

        x0_score = x_score + (sigma_t ** 2) * score_score
        return x0_score, score_score, sigma_t

    def _pad_to_model_size(self, x: torch.Tensor):
        target = int(self.config.data.image_size)
        _, _, h, w = x.shape

        pad_h = max(0, target - h)
        pad_w = max(0, target - w)

        # 오른쪽/아래만 pad
        x_pad = F.pad(x, (0, pad_w, 0, pad_h), mode="replicate")

        meta = {
            "orig_h": h,
            "orig_w": w,
            "target": target,
            "pad_h": pad_h,
            "pad_w": pad_w,
        }
        return x_pad, meta

    def _crop_back(self, x: torch.Tensor, meta: dict):
        return x[..., : meta["orig_h"], : meta["orig_w"]]

    def reconstruct(
        self,
        x_init: torch.Tensor,
        y: torch.Tensor,
        save_path: Path,
        cg_iter: int,
        w_dz: float = 0.5,
        noise_control: Optional[Dict] = None,
        use_init: bool = False,
        save_intermediates: bool = False,
        renoise_method: str = "DDPM",
    ) -> torch.Tensor:
        sde = self.sde
        eps = self.eps

        with torch.no_grad():
            timesteps = torch.linspace(sde.T, eps, sde.N)

            if use_init:
                noises = torch.randn_like(x_init).to(y.device)

                t = timesteps[0]
                x_t = autils.re_noising(
                    renoise_method="DDPM",
                    x_0_t=x_init,
                    score_t=None,
                    noises=noises,
                    sde=sde,
                    t_curr=t,
                    t_next=t,
                    noise_control=noise_control,
                )
            else:
                x_t = sde.prior_sampling(x_init.shape).to(y.device)

            pbar = tqdm(total=sde.N, desc="(0. DM)", colour="red", leave=True)

            for i in range(sde.N):
                t = timesteps[i]

                x_score = self._vol_to_score(x_t)
                x_score_batch = autils.batchfy(x_score, 20)
                x0_score_batches = []
                score_score_batches = []
                sigma_t = torch.tensor(0.0, device=x_t.device)

                pbar_batch = tqdm(total=len(x_score_batch), desc="(1. Batch)", colour="green", position=1, leave=False)

                for x_score_b in x_score_batch:
                    x0_score_b, score_score_b, sigma_t = self.tweedie_denoising_fn(x_score_b, t)
                    x0_score_batches.append(x0_score_b)
                    score_score_batches.append(score_score_b)

                    pbar_batch.update()

                x_0_t = torch.cat(x0_score_batches, dim=0)
                score_t = torch.cat(score_score_batches, dim=0)

                pbar.set_postfix({"t": i, "sigma_t": float(sigma_t.item())})
                pbar.update()

                if save_intermediates:
                    save_nii_image(x_0_t, f"{save_path}/X_0/x_0_{i:04d}.nii.gz")

                ATy = self._AT(y)

                x_0_t_hat = autils.cg_uni(self.A_cg_tik, ATy, x_0_t, rho=0, maxiter=cg_iter, tol=1e-8)

                if i == sde.N - 1:
                    x_t = x_0_t_hat
                    break

                noises = torch.randn_like(x_0_t_hat).to(y.device)

                x_t = autils.re_noising(
                    renoise_method=renoise_method,
                    x_0_t=x_0_t_hat,
                    score_t=score_t,
                    noises=noises,
                    sde=sde,
                    t_curr=timesteps[i],
                    t_next=timesteps[i + 1],
                    noise_control=noise_control,
                )

                if save_intermediates:
                    save_nii_image(x_t, f"{save_path}/X_hat/Re-noising/x_{i:04d}_addNoise.nii.gz")

        return x_t
