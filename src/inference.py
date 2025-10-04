import argparse
import os
import pathlib
import random
from datetime import datetime
from types import SimpleNamespace

import SimpleITK as sitk
import numpy as np
import torch
import torch.optim
import torch.utils.data
import yaml
import torch.nn.functional as F
from torch.amp import autocast

from src import models
from src.dataset import get_datasets
from src.dataset.batch_utils import pad_batch1_to_compatible_size
from src.models import get_norm_layer
from src.tta import apply_simple_tta
from src.utils import reload_ckpt_bis

parser = argparse.ArgumentParser(description='Brats validation and testing dataset inference')
parser.add_argument('--config', default='', type=str, metavar='PATH',
                    help='path(s) to the trained models config yaml you want to use', nargs="+")
parser.add_argument('--devices', required=True, type=str,
                    help='Set the CUDA_VISIBLE_DEVICES env var from this string')
parser.add_argument('--on', default="val", choices=["val","train","test"])
parser.add_argument('--tta', action="store_true")
parser.add_argument('--seed', default=16111990)


def main(args):
    # setup
    random.seed(args.seed)
    ngpus = torch.cuda.device_count()
    if ngpus == 0:
        raise RuntimeWarning("This will not be able to run on CPU only")
    print(f"Working with {ngpus} GPUs")
    print(args.config)

    current_experiment_time = datetime.now().strftime('%Y%m%d_%T').replace(":", "")
    save_folder = pathlib.Path(f"./preds/{current_experiment_time}")
    save_folder.mkdir(parents=True, exist_ok=True)

    with (save_folder / 'args.txt').open('w') as f:
        print(vars(args), file=f)

    args_list = []
    for config in args.config:
        config_file = pathlib.Path(config).resolve()
        print(config_file)
        ckpt = config_file.with_name("model_best.pth.tar")
        with config_file.open("r") as file:
            old_args = yaml.safe_load(file)
            old_args = SimpleNamespace(**old_args, ckpt=ckpt)
            # set default normalisation
            if not hasattr(old_args, "normalisation"):
                old_args.normalisation = "minmax"
        print(old_args)
        args_list.append(old_args)

    if args.on == "test":
        args.pred_folder = save_folder / f"test_segs_tta{args.tta}"
        args.pred_folder.mkdir(exist_ok=True)
    elif args.on == "val":
        args.pred_folder = save_folder / f"validation_segs_tta{args.tta}"
        args.pred_folder.mkdir(exist_ok=True)
    else:
        args.pred_folder = save_folder / f"training_segs_tta{args.tta}"
        args.pred_folder.mkdir(exist_ok=True)

    # Create model
    models_list = []
    normalisations_list = []
    for model_args in args_list:
        print(model_args.arch)
        model_maker = getattr(models, model_args.arch)

        model = model_maker(
            4, 3,
            width=model_args.width, deep_supervision=model_args.deep_sup,
            norm_layer=get_norm_layer(model_args.norm_layer), dropout=model_args.dropout)
        print(f"Creating {model_args.arch}")

        reload_ckpt_bis(str(model_args.ckpt), model)
        model.eval()  # ensure eval mode for inference
        models_list.append(model)
        normalisations_list.append(model_args.normalisation)
        print("reload best weights")
        print(model)

    dataset_minmax = get_datasets(args.seed, False, no_seg=True,
                                  on=args.on, normalisation="minmax")

    dataset_zscore = get_datasets(args.seed, False, no_seg=True,
                                  on=args.on, normalisation="zscore")

    loader_minmax = torch.utils.data.DataLoader(
        dataset_minmax, batch_size=1, num_workers=2)

    loader_zscore = torch.utils.data.DataLoader(
        dataset_zscore, batch_size=1, num_workers=2)

    print("Val dataset number of batch:", len(loader_minmax))
    generate_segmentations((loader_minmax, loader_zscore), models_list, normalisations_list, args)


def center_crop_to_size(tensor, target_size):
    # tensor: (N, C, Z, Y, X)
    _, _, z, y, x = tensor.shape
    start_z = (z - target_size) // 2 if z > target_size else 0
    start_y = (y - target_size) // 2 if y > target_size else 0
    start_x = (x - target_size) // 2 if x > target_size else 0
    end_z = start_z + min(z, target_size)
    end_y = start_y + min(y, target_size)
    end_x = start_x + min(x, target_size)
    tensor = tensor[:, :, start_z:end_z, start_y:end_y, start_x:end_x]
    return tensor, (start_z, start_y, start_x)


def _get_crop_start(crops_idx, dim_idx):
    """
    Lấy start index theo chiều dim_idx từ crop_indexes.
    Hỗ trợ dạng:
      - list/tuple: crops_idx[dim_idx][0] là tensor hoặc số
      - tensor (3,2) hoặc (1,3,2)
    """
    if torch.is_tensor(crops_idx):
        t = crops_idx
        if t.dim() == 3 and t.size(0) == 1:
            t = t[0]
        # expect shape (3,2): index 0 is start
        return int(t[dim_idx, 0].item())
    # list/tuple
    v = crops_idx[dim_idx][0]
    return int(v.item()) if torch.is_tensor(v) else int(v)


def generate_segmentations(data_loaders, models_list, normalisations, args):
    target_size = 128
    canvas_dhw = (155, 240, 240)  # (Z, Y, X) of BRATS full volume

    for i, (batch_minmax, batch_zscore) in enumerate(zip(data_loaders[0], data_loaders[1])):
        patient_id = batch_minmax["patient_id"][0]
        ref_img_path = batch_minmax["seg_path"][0]

        crops_idx_minmax = batch_minmax["crop_indexes"]
        crops_idx_zscore = batch_zscore["crop_indexes"]
        inputs_minmax = batch_minmax["image"]
        inputs_zscore = batch_zscore["image"]

        # Center crop nhưng KHÔNG sửa crop_indexes in-place
        inputs_minmax, starts_minmax = center_crop_to_size(inputs_minmax, target_size)
        inputs_zscore, starts_zscore = center_crop_to_size(inputs_zscore, target_size)

        inputs_minmax, pads_minmax = pad_batch1_to_compatible_size(inputs_minmax)
        inputs_zscore, pads_zscore = pad_batch1_to_compatible_size(inputs_zscore)

        model_preds = []

        for model, normalisation in zip(models_list, normalisations):
            if normalisation == "minmax":
                inputs = inputs_minmax.cuda(non_blocking=True)
                pads = pads_minmax
                crops_idx = crops_idx_minmax
                starts = starts_minmax
            elif normalisation == "zscore":
                inputs = inputs_zscore.cuda(non_blocking=True)
                pads = pads_zscore
                crops_idx = crops_idx_zscore
                starts = starts_zscore
            else:
                raise ValueError(f"Unknown normalisation: {normalisation}")

            # Tính start index mới: start_gốc + offset center-crop
            z0 = _get_crop_start(crops_idx, 0) + int(starts[0])
            y0 = _get_crop_start(crops_idx, 1) + int(starts[1])
            x0 = _get_crop_start(crops_idx, 2) + int(starts[2])

            model = model.cuda()
            with autocast("cuda"):
                with torch.no_grad():
                    if args.tta:
                        pre_segs = apply_simple_tta(model, inputs, average=True)
                        # Đảm bảo là prob
                        if pre_segs.min() < 0 or pre_segs.max() > 1:
                            pre_segs = torch.sigmoid(pre_segs)
                    else:
                        if getattr(model, "deep_supervision", False):
                            pre_segs, _ = model(inputs)
                        else:
                            pre_segs = model(inputs)
                        pre_segs = torch.sigmoid(pre_segs)

                    # Bỏ pads
                    maxz = pre_segs.size(2) - pads[0]
                    maxy = pre_segs.size(3) - pads[1]
                    maxx = pre_segs.size(4) - pads[2]
                    pre_segs = pre_segs[:, :, 0:maxz, 0:maxy, 0:maxx].cpu()

            # Canvas (1, 3, 155, 240, 240)
            segs = torch.zeros((1, 3, *canvas_dhw), dtype=pre_segs.dtype)

            # Kích thước nguồn
            sz, sy, sx = pre_segs.size(2), pre_segs.size(3), pre_segs.size(4)

            # Clamp đích theo canvas
            z1 = min(z0 + sz, canvas_dhw[0])
            y1 = min(y0 + sy, canvas_dhw[1])
            x1 = min(x0 + sx, canvas_dhw[2])

            # Số voxel thực sự chèn được
            ins_z = max(0, z1 - z0)
            ins_y = max(0, y1 - y0)
            ins_x = max(0, x1 - x0)

            if ins_z == 0 or ins_y == 0 or ins_x == 0:
                # Nếu có gì đó bất thường khiến ROI nằm ngoài canvas, bỏ qua model này
                print(f"Warning: empty insertion for patient {patient_id} (z0,y0,x0)=({z0},{y0},{x0}), size=({sz},{sy},{sx})")
                model = model.cpu()
                continue

            # Cắt nguồn tương ứng nếu cần
            src_z1 = min(ins_z, sz)
            src_y1 = min(ins_y, sy)
            src_x1 = min(ins_x, sx)

            # Gán
            segs[0, :, z0:z0+ins_z, y0:y0+ins_y, x0:x0+ins_x] = pre_segs[0, :, 0:src_z1, 0:src_y1, 0:src_x1]
            model = model.cpu()
            model_preds.append(segs)

        # Ensemble theo model
        if len(model_preds) == 0:
            # Không có dự đoán hợp lệ
            print(f"Warning: no predictions for patient {patient_id}, writing empty mask.")
            pre_segs_mean = torch.zeros((1, 3, *canvas_dhw))
        else:
            pre_segs_mean = torch.stack(model_preds, dim=0).mean(dim=0)

        segs_bin = pre_segs_mean[0].numpy() > 0.5  # (3, Z, Y, X)

        et = segs_bin[0]
        net = np.logical_and(segs_bin[1], np.logical_not(et))
        ed = np.logical_and(segs_bin[2], np.logical_not(segs_bin[1]))
        labelmap_np = np.zeros(segs_bin[0].shape, dtype=np.uint8)
        labelmap_np[et] = 4
        labelmap_np[net] = 1
        labelmap_np[ed] = 2

        # SITK: array (D,H,W) -> image (W,H,D)
        labelmap = sitk.GetImageFromArray(labelmap_np)
        ref_img = sitk.ReadImage(ref_img_path)

        # Nếu size không khớp (do hướng trục khác), thử transpose
        if labelmap.GetSize() != ref_img.GetSize():
            labelmap_np_transposed = labelmap_np.transpose(2, 1, 0)
            labelmap = sitk.GetImageFromArray(labelmap_np_transposed)

        labelmap.CopyInformation(ref_img)
        out_path = f"{str(args.pred_folder)}/{patient_id}.nii.gz"
        print(f"Writing {out_path}")
        sitk.WriteImage(labelmap, out_path)


if __name__ == '__main__':
    arguments = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = arguments.devices
    main(arguments)