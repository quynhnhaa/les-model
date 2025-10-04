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
from torch.cuda.amp import autocast

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
    c, z, y, x = tensor.shape[1:]
    start_z = (z - target_size) // 2 if z > target_size else 0
    start_y = (y - target_size) // 2 if y > target_size else 0
    start_x = (x - target_size) // 2 if x > target_size else 0
    end_z = start_z + min(z, target_size)
    end_y = start_y + min(y, target_size)
    end_x = start_x + min(x, target_size)
    tensor = tensor[:, :, start_z:end_z, start_y:end_y, start_x:end_x]
    return tensor, (start_z, start_y, start_x)


def generate_segmentations(data_loaders, models, normalisations, args):
    # TODO: try reuse the function used for train...
    target_size = 128
    for i, (batch_minmax, batch_zscore) in enumerate(zip(data_loaders[0], data_loaders[1])):
        patient_id = batch_minmax["patient_id"][0]
        ref_img_path = batch_minmax["seg_path"][0]
        crops_idx_minmax = batch_minmax["crop_indexes"]
        crops_idx_zscore = batch_zscore["crop_indexes"]
        inputs_minmax = batch_minmax["image"]
        inputs_zscore = batch_zscore["image"]

        # Center crop to target_size if larger
        inputs_minmax, starts_minmax = center_crop_to_size(inputs_minmax, target_size)
        inputs_zscore, starts_zscore = center_crop_to_size(inputs_zscore, target_size)

        # Update crop_indexes
        crops_idx_minmax[0][0] += starts_minmax[0]
        crops_idx_minmax[1][0] += starts_minmax[1]
        crops_idx_minmax[2][0] += starts_minmax[2]
        if inputs_minmax.shape[2] == target_size:
            crops_idx_minmax[0][1] = crops_idx_minmax[0][0] + target_size
        if inputs_minmax.shape[3] == target_size:
            crops_idx_minmax[1][1] = crops_idx_minmax[1][0] + target_size
        if inputs_minmax.shape[4] == target_size:
            crops_idx_minmax[2][1] = crops_idx_minmax[2][0] + target_size

        crops_idx_zscore[0][0] += starts_zscore[0]
        crops_idx_zscore[1][0] += starts_zscore[1]
        crops_idx_zscore[2][0] += starts_zscore[2]
        if inputs_zscore.shape[2] == target_size:
            crops_idx_zscore[0][1] = crops_idx_zscore[0][0] + target_size
        if inputs_zscore.shape[3] == target_size:
            crops_idx_zscore[1][1] = crops_idx_zscore[1][0] + target_size
        if inputs_zscore.shape[4] == target_size:
            crops_idx_zscore[2][1] = crops_idx_zscore[2][0] + target_size

        inputs_minmax, pads_minmax = pad_batch1_to_compatible_size(inputs_minmax)
        inputs_zscore, pads_zscore = pad_batch1_to_compatible_size(inputs_zscore)
        model_preds = []
        last_norm = None
        for model, normalisation in zip(models, normalisations):
            if normalisation == last_norm:
                pass
            elif normalisation == "minmax":
                inputs = inputs_minmax.cuda()
                pads = pads_minmax
                crops_idx = crops_idx_minmax
            elif normalisation == "zscore":
                inputs = inputs_zscore.cuda()
                pads = pads_zscore
                crops_idx = crops_idx_zscore
            model.cuda()  # go to gpu
            with autocast():
                with torch.no_grad():
                    if args.tta:
                        pre_segs = apply_simple_tta(model, inputs, True)
                        model_preds.append(pre_segs)
                    else:
                        if model.deep_supervision:
                            pre_segs, _ = model(inputs)
                        else:
                            pre_segs = model(inputs)
                        pre_segs = pre_segs.sigmoid_().cpu()
                    # remove pads
                    maxz, maxy, maxx = pre_segs.size(2) - pads[0], pre_segs.size(3) - pads[1], pre_segs.size(4) - \
                                       pads[2]
                    pre_segs = pre_segs[:, :, 0:maxz, 0:maxy, 0:maxx].cpu()
                    print("pre_segs size", pre_segs.shape)
                    segs = torch.zeros((1, 3, 155, 240, 240))

                    # Correct indexing based on debug output: crops_idx is a list of lists of tensors
                    z_slice = slice(crops_idx[0][0].item(), crops_idx[0][1].item())
                    y_slice = slice(crops_idx[1][0].item(), crops_idx[1][1].item())
                    x_slice = slice(crops_idx[2][0].item(), crops_idx[2][1].item())

                    segs[0, :, z_slice, y_slice, x_slice] = pre_segs[0]
                    print("segs size", segs.shape)

                    model_preds.append(segs)
            model.cpu()  # free for the next one
        pre_segs = torch.stack(model_preds).mean(dim=0)

        segs = pre_segs[0].numpy() > 0.5

        et = segs[0]
        net = np.logical_and(segs[1], np.logical_not(et))
        ed = np.logical_and(segs[2], np.logical_not(segs[1]))
        labelmap_np = np.zeros(segs[0].shape, dtype=np.uint8)
        labelmap_np[et] = 4
        labelmap_np[net] = 1
        labelmap_np[ed] = 2

        # The numpy array has shape (D, H, W), e.g. (155, 240, 240)
        # sitk.GetImageFromArray creates an image of size (W, H, D)
        labelmap = sitk.GetImageFromArray(labelmap_np)

        ref_img = sitk.ReadImage(ref_img_path)

        # The error indicates that for some data, ref_img has a permuted size.
        # We check for this and transpose our numpy array before creating the
        # sitk image to match the reference orientation.
        if labelmap.GetSize() != ref_img.GetSize():
            # Transpose the numpy array from (D, H, W) to (W, H, D)
            labelmap_np_transposed = labelmap_np.transpose(2, 1, 0)
            labelmap = sitk.GetImageFromArray(labelmap_np_transposed)

        labelmap.CopyInformation(ref_img)
        print(f"Writing {str(args.pred_folder)}/{patient_id}.nii.gz")
        sitk.WriteImage(labelmap, f"{str(args.pred_folder)}/{patient_id}.nii.gz")


if __name__ == '__main__':
    arguments = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = arguments.devices
    main(arguments)
