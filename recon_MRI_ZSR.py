import datetime
import os
from pathlib import Path

import SimpleITK as sitk
import torch
import yaml
from ml_collections import config_dict
from torchvision import transforms
import torch.nn.functional as F

import models.ncsnpp
import utils.args
import utils.data
import utils.model
import utils.result
from algorithms.MRI_ZSR import DDNM, DDS
from physics import zsr
from torchvision.utils import save_image, make_grid

torch.set_num_threads(20)
HU_MAX = 440
HU_MIN = 20

def _crop_to_common_volume(pred: torch.Tensor, gt: torch.Tensor):
    """
    pred, gt: [Z, C, Y, X]
    Z/Y/X 공통 영역으로 crop
    """
    if pred.ndim != 4 or gt.ndim != 4:
        raise ValueError(f"Expected 4D tensors [Z,C,Y,X], got {pred.shape} and {gt.shape}")

    if pred.shape[1] != gt.shape[1]:
        raise ValueError(f"Channel mismatch: pred {pred.shape}, gt {gt.shape}")

    z = min(pred.shape[0], gt.shape[0])
    y = min(pred.shape[2], gt.shape[2])
    x = min(pred.shape[3], gt.shape[3])

    pred_crop = pred[:z, :, :y, :x]
    gt_crop = gt[:z, :, :y, :x]
    return pred_crop, gt_crop


def _pad_pred_to_gt_volume(pred: torch.Tensor, gt: torch.Tensor):
    """
    pred를 gt의 [Z,C,Y,X] shape에 맞춰 pad
    Z는 마지막 slice 복제로 pad
    Y/X는 replicate pad
    """
    if pred.ndim != 4 or gt.ndim != 4:
        raise ValueError(f"Expected 4D tensors [Z,C,Y,X], got {pred.shape} and {gt.shape}")

    if pred.shape[1] != gt.shape[1]:
        raise ValueError(f"Channel mismatch: pred {pred.shape}, gt {gt.shape}")

    z_pad = gt.shape[0] - pred.shape[0]
    y_pad = gt.shape[2] - pred.shape[2]
    x_pad = gt.shape[3] - pred.shape[3]

    if z_pad < 0 or y_pad < 0 or x_pad < 0:
        raise ValueError(
            f"pred shape {tuple(pred.shape)} is larger than gt shape {tuple(gt.shape)}; "
            "cannot pad pred to gt."
        )

    out = pred

    # 먼저 Y/X pad
    if y_pad > 0 or x_pad > 0:
        # [Z,C,Y,X] -> F.pad pads last dims only
        out = F.pad(out, (0, x_pad, 0, y_pad), mode="replicate")

    # 그 다음 Z pad
    if z_pad > 0:
        last_slice = out[-1:].repeat(z_pad, 1, 1, 1)
        out = torch.cat([out, last_slice], dim=0)

    return out

def get_orientation_code(img: sitk.Image) -> str:
    # Obtain the direction string (e.g., 'RAI', 'LPI', etc.);
    # note that the syntax varies slightly across different versions of SimpleITK.
    f = sitk.DICOMOrientImageFilter()
    return f.GetOrientationFromDirectionCosines(img.GetDirection())

def volume_to_score_batch(vol: torch.Tensor, plane: str) -> torch.Tensor:
    """
    vol: [Z, 1, Y, X]
    return score-model batch [B, 1, H, W]
    """
    if plane == "axial":
        # [Z, 1, Y, X]
        return vol
    if plane == "coronal":
        # batch over Y -> [Y, 1, Z, X]
        return vol.permute(2, 1, 0, 3)
    if plane == "sagittal":
        # batch over X -> [X, 1, Z, Y]
        return vol.permute(3, 1, 0, 2)
    raise ValueError(f"Unknown plane: {plane}")


def score_batch_to_volume(x: torch.Tensor, plane: str) -> torch.Tensor:
    """
    x: [B, 1, H, W]
    return canonical volume [Z, 1, Y, X]
    """
    if plane == "axial":
        return x
    if plane == "coronal":
        # [Y, 1, Z, X] -> [Z, 1, Y, X]
        return x.permute(2, 1, 0, 3)
    if plane == "sagittal":
        # [X, 1, Z, Y] -> [Z, 1, Y, X]
        return x.permute(2, 1, 3, 0)
    raise ValueError(f"Unknown plane: {plane}")


def volume_to_physics_batch(vol: torch.Tensor) -> torch.Tensor:
    """
    canonical volume [Z, 1, Y, X]
    -> current ZSR physics batch [X, 1, Y, Z]
    마지막 축이 항상 z가 되도록 유지
    """
    return vol.permute(3, 1, 2, 0)


def physics_batch_to_volume(x: torch.Tensor) -> torch.Tensor:
    """
    physics batch [X, 1, Y, Z]
    -> canonical volume [Z, 1, Y, X]
    """
    return x.permute(3, 1, 2, 0)


def load_and_preprocess_image(args, config):
    gt_sitk = sitk.ReadImage(args.data)
    gt_image = sitk.GetArrayViewFromImage(gt_sitk)  # [Z, Y, X]

    D = gt_image.shape[0]
    if args.slice_begin == 0 and args.slice_end == 0:
        args.slice_begin, args.slice_end = 0, D
    args.slice_begin = max(0, args.slice_begin)
    args.slice_end = min(D, args.slice_end)

    gt_image = gt_image[args.slice_begin : args.slice_end]  # [Z, Y, X]
    gt_image = torch.tensor(gt_image, dtype=torch.float32, device=config.device).unsqueeze(1)  # [Z,1,Y,X]

    gt_image = utils.data.HU_to_norm_01(
        gt_image, gt_image.max().item(), gt_image.min().item()
    )

    # score model이 보는 plane 기준으로만 resize 검사
    x_score = volume_to_score_batch(gt_image, args.plane)
    target_size = int(config.data.image_size)
    if x_score.shape[-2:] != (target_size, target_size):
        x_score = F.interpolate(
            x_score,
            size=(target_size, target_size),
            mode="bilinear",
            align_corners=False,
        )
        gt_image = score_batch_to_volume(x_score, args.plane)

    print(f"[INFO] plane = {args.plane}", flush=True)
    print(f"[INFO] config.data.image_size = {target_size}", flush=True)
    print(f"[INFO] gt_volume shape = {tuple(gt_image.shape)}", flush=True)
    print(f"[INFO] score-batch shape = {tuple(volume_to_score_batch(gt_image, args.plane).shape)}", flush=True)
    return gt_image


def main(args):
    # Load and Setting Configs
    # ----------------------------------
    config_path = args.config_path
    config_name = config_path.split("/")[-1].replace(".yaml", "")
    with open(config_path, "r") as f:
        config_data = yaml.safe_load(f)

    config = config_dict.ConfigDict(config_data)
    config.device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    model_image_size = int(config.data.image_size)

    if getattr(args, "recon_size", 0) in [0, None]:
        args.recon_size = model_image_size
    elif int(args.recon_size) != model_image_size:
        print(
            f"[WARN] args.recon_size={args.recon_size} but config.data.image_size={model_image_size}. "
            f"Using config.data.image_size={model_image_size} for model compatibility.",
            flush=True,
        )
        args.recon_size = model_image_size

    print(f"[INFO] model image_size: {model_image_size}", flush=True)
    print(f"[INFO] effective recon_size: {args.recon_size}", flush=True)

    data_path = Path(args.data)
    data_name = data_path.stem

    # Build problem string and include weight fields only when they are non-zero
    parts = [
        f"{args.task}",
        f"{args.degree}",
        f"{args.slice_begin:04d}",
        f"{args.slice_end:04d}",
        f"{args.NFE}",
        f"{args.use_init}",
    ]
    # Only include regularization weights when they are non-zero to keep names concise
    if getattr(args, "num_cg", 0):
        parts.append(f"nCG-{args.num_cg}")
    if getattr(args, "w_dps", 0):
        parts.append(f"wDPS-{args.w_dps}")
    if getattr(args, "w_tik", 0):
        parts.append(f"wTIK-{args.w_tik}")
    if getattr(args, "w_dz", 0):
        parts.append(f"wDZ-{args.w_dz}")

    # Always include noise_control and renoise_method (may be None)
    parts.append(str(args.noise_control))
    parts.append(str(args.renoise_method))

    problem = "_".join(map(str, parts))

    print(f"Task: {problem}")

    # Data
    gt_image = load_and_preprocess_image(args, config)  # [0,1]
    if gt_image.shape[-2:] != (config.data.image_size, config.data.image_size):
        raise ValueError(
            f"Input to score model must match config.data.image_size={(config.data.image_size, config.data.image_size)}, "
            f"but got {tuple(gt_image.shape[-2:])}"
        )

    measure_model = zsr.ZAxisSuperResolution(args.degree)
    gt_phys = volume_to_physics_batch(gt_image)  # [X,1,Y,Z]
    y = measure_model.A(gt_phys).float().to(config.device).clone().detach()
    y = torch.tensor(y.cpu().numpy()).to(config.device)

    ATy_phys = measure_model.A_dagger(y).clone().detach()
    ATy_phys = measure_model.couple(measure_model.decouple(ATy_phys))
    ATy = physics_batch_to_volume(ATy_phys)  # back to canonical [Z,1,Y,X]

    # Create Save Path
    # ----------------------------------
    save_root = Path(
        f"/data/jionkim/ISCS-main/results/{config_name}/{data_name}/{args.task}-{args.degree}/{args.method}/{problem}/{datetime.datetime.now():%y%m%d_%H%M%S}/"
    )
    save_root.mkdir(parents=True, exist_ok=True)

    utils.result.save_nii_image(y, os.path.join(save_root, "y.nii.gz"))
    utils.result.save_nii_image(ATy, os.path.join(save_root, "ATy.nii.gz"))
    utils.result.save_nii_image(gt_image, os.path.join(save_root, "GT.nii.gz"))

    diffusion_methods = ["DDS", "DDNM", "uncondition"]
    if args.method in diffusion_methods:
        # Setup Diffusion Model
        # ----------------------------------
        ckpt_filename = args.checkpoint_path
        from sde_lib import VESDE

        sde = VESDE(sigma_min=args.sigma_min, sigma_max=args.sigma_max, N=args.NFE)
        score_model = utils.model.get_score_models(config, ckpt_filename)

        # Create Pipeline
        # ----------------------------------
        if args.method == "DDS":
            print("Run DDS!")
            recon_pipeline = DDS.DDS(sde, score_model, config=config, measure_model=measure_model, plane="axial")
            x = recon_pipeline.reconstruct(
                x_init=ATy,
                y=y,
                cg_iter=args.num_cg,
                w_dz=args.w_dz,
                save_path=save_root,
                use_init=args.use_init,
                save_intermediates=True,
                noise_control=args.noise_control,
                renoise_method=args.renoise_method,
            )
        elif args.method == "DDNM":
            print("Run DDNM!")
            recon_pipeline = DDNM.DDNMReconstructor(
                sde, score_model, config=config, measure_model=measure_model, factor=args.degree
            )
            x = recon_pipeline.reconstruct(
                x_init=ATy,
                y=y,
                save_path=save_root,
                use_init=args.use_init,
                save_intermediates=True,
                noise_control=args.noise_control,
            )
        elif args.method == "uncondition":
            print("Run un-condition generation!")
    else:
        raise ValueError("Invalid method")
    # * ---------------------------------------

    data_range_gt = (gt_image.max() - gt_image.min()).item()

    print(f"[INFO] raw pred shape : {tuple(x.shape)}", flush=True)
    print(f"[INFO] metric gt shape: {tuple(gt_image.shape)}", flush=True)

    # 1) metric용: 공통 영역 crop
    x_eval, gt_eval = _crop_to_common_volume(x, gt_image)

    print(f"[INFO] eval pred shape: {tuple(x_eval.shape)}", flush=True)
    print(f"[INFO] eval gt shape  : {tuple(gt_eval.shape)}", flush=True)

    # 2) 저장용: raw / padded 둘 다 저장
    utils.result.save_nii_image(x, os.path.join(save_root, "recon_raw.nii.gz"))

    x_save = _pad_pred_to_gt_volume(x, gt_image)
    utils.result.save_nii_image(x_save, os.path.join(save_root, "recon_padded_to_gt.nii.gz"))

    psnr, ssim = utils.result.cal_metrics(x_eval, gt_eval, save_root)

    print(f"Save to: {save_root}")
    print(f"PSNR: {psnr}\nSSIM: {ssim}")

    metrics = utils.result.compute_slice_metrics_optimized(x_eval, gt_eval, data_range=data_range_gt)
    print(f"{'View':<10}{'PSNR':>12}{'SSIM':>12}")
    print(f"{'Axial':<10}{metrics['axial']['PSNR_mean']:12.4f}{metrics['axial']['SSIM_mean']:12.4f}")
    print(f"{'Coronal':<10}{metrics['coronal']['PSNR_mean']:12.4f}{metrics['coronal']['SSIM_mean']:12.4f}")
    print(f"{'Sagittal':<10}{metrics['sagittal']['PSNR_mean']:12.4f}{metrics['sagittal']['SSIM_mean']:12.4f}")


if __name__ == "__main__":
    args = utils.args.build_parser()
    main(args)
