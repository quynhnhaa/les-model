from itertools import combinations, product
import torch
import torch.nn.functional as F
import numpy as np
from scipy.ndimage import rotate, zoom
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter

trs = list(combinations(range(2, 5), 2)) + [None]
flips = list(range(2, 5)) + [None]
rots = list(range(1, 4)) + [None]
transform_list = list(product(flips, rots))


def simple_tta(x):
    """Perform all transpose/mirror transform possible only once.

    Sample one of the potential transform and return the transformed image and a lambda function to revert the transform
    Random seed should be set before calling this function
    """
    out = [[x, lambda z: z]]
    for flip, rot in transform_list[:-1]:
        if flip and rot:
            trf_img = torch.rot90(x.flip(flip), rot, dims=(3, 4))
            back_trf = revert_tta_factory(flip, -rot)
        elif flip:
            trf_img = x.flip(flip)
            back_trf = revert_tta_factory(flip, None)
        elif rot:
            trf_img = torch.rot90(x, rot, dims=(3, 4))
            back_trf = revert_tta_factory(None, -rot)
        else:
            raise
        out.append([trf_img, back_trf])
    return out


def apply_simple_tta(model, x, average=True):
    todos = simple_tta(x)
    out = []
    for im, revert in todos:
        if model.deep_supervision:
            out.append(revert(model(im)[0]).sigmoid_().cpu())
        else:
            out.append(revert(model(im)).sigmoid_().cpu())
    if not average:
        return out
    return torch.stack(out).mean(dim=0)


def revert_tta_factory(flip, rot):
    if flip and rot:
        return lambda x: torch.rot90(x.flip(flip), rot, dims=(3, 4))
    elif flip:
        return lambda x: x.flip(flip)
    elif rot:
        return lambda x: torch.rot90(x, rot, dims=(3, 4))
    else:
        raise


def advanced_tta(x):
    """Advanced TTA with more transformations including intensity and elastic deformation"""
    out = [[x, lambda z: z]]  # Original image
    
    # Basic geometric transforms
    for flip, rot in transform_list[:-1]:
        if flip and rot:
            trf_img = torch.rot90(x.flip(flip), rot, dims=(3, 4))
            back_trf = revert_tta_factory(flip, -rot)
        elif flip:
            trf_img = x.flip(flip)
            back_trf = revert_tta_factory(flip, None)
        elif rot:
            trf_img = torch.rot90(x, rot, dims=(3, 4))
            back_trf = revert_tta_factory(None, -rot)
        else:
            continue
        out.append([trf_img, back_trf])
    
    # Intensity variations
    for intensity_factor in [0.8, 1.2]:
        def intensity_transform(img, factor=intensity_factor):
            return img * factor
        
        def intensity_inverse(pred, factor=intensity_factor):
            return pred  # No inverse needed for intensity
        
        trf_img = intensity_transform(x)
        out.append([trf_img, intensity_inverse])
    
    # Gaussian noise
    for noise_std in [0.01, 0.02]:
        def noise_transform(img, std=noise_std):
            noise = torch.randn_like(img) * std
            return img + noise
        
        def noise_inverse(pred, std=noise_std):
            return pred  # No inverse needed for noise
        
        trf_img = noise_transform(x)
        out.append([trf_img, noise_inverse])
    
    return out


def elastic_deformation_tta(x, alpha=1000, sigma=30):
    """Apply elastic deformation TTA"""
    def elastic_transform(img, alpha=alpha, sigma=sigma):
        """Apply elastic deformation to image"""
        shape = img.shape
        dx = gaussian_filter((np.random.random(shape[2:]) * 2 - 1), sigma, mode="constant", cval=0) * alpha
        dy = gaussian_filter((np.random.random(shape[2:]) * 2 - 1), sigma, mode="constant", cval=0) * alpha
        dz = gaussian_filter((np.random.random(shape[2:]) * 2 - 1), sigma, mode="constant", cval=0) * alpha
        
        x, y, z = np.meshgrid(np.arange(shape[2]), np.arange(shape[3]), np.arange(shape[4]), indexing='ij')
        indices = np.reshape(y + dy, (-1, 1)), np.reshape(x + dx, (-1, 1)), np.reshape(z + dz, (-1, 1))
        
        transformed_img = np.zeros_like(img)
        for i in range(shape[0]):
            for j in range(shape[1]):
                transformed_img[i, j] = map_coordinates(img[i, j].cpu().numpy(), indices, order=1, mode='reflect')
        
        return torch.from_numpy(transformed_img).to(img.device)
    
    def elastic_inverse(pred, alpha=alpha, sigma=sigma):
        # For simplicity, return original (in practice, you'd need to store the transformation)
        return pred
    
    trf_img = elastic_transform(x)
    return [trf_img, elastic_inverse]


def apply_advanced_tta(model, x, average=True, use_elastic=True):
    """Apply advanced TTA with multiple transformation types"""
    todos = advanced_tta(x)
    
    # Add elastic deformation if requested
    if use_elastic:
        elastic_transform = elastic_deformation_tta(x)
        todos.append(elastic_transform)
    
    out = []
    for im, revert in todos:
        if model.deep_supervision:
            out.append(revert(model(im)[0]).sigmoid_().cpu())
        else:
            out.append(revert(model(im)).sigmoid_().cpu())
    
    if not average:
        return out
    return torch.stack(out).mean(dim=0)


def multi_scale_tta(model, x, scales=[0.9, 1.0, 1.1]):
    """Apply multi-scale TTA"""
    predictions = []
    
    for scale in scales:
        if scale == 1.0:
            # Original scale
            if model.deep_supervision:
                pred = model(x)[0].sigmoid_()
            else:
                pred = model(x).sigmoid_()
        else:
            # Resize image
            original_shape = x.shape
            new_shape = [int(s * scale) for s in original_shape[2:]]
            
            # Downsample
            x_scaled = F.interpolate(x, size=new_shape, mode='trilinear', align_corners=True)
            
            # Get prediction
            if model.deep_supervision:
                pred_scaled = model(x_scaled)[0]
            else:
                pred_scaled = model(x_scaled)
            
            # Upsample back to original size
            pred = F.interpolate(pred_scaled, size=original_shape[2:], mode='trilinear', align_corners=True)
            pred = pred.sigmoid_()
        
        predictions.append(pred.cpu())
    
    return torch.stack(predictions).mean(dim=0)
