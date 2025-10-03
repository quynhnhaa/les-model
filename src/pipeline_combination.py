"""Intelligent combination of Pipeline A and Pipeline B results"""
import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import nibabel as nib
from scipy import ndimage
from skimage.measure import label, regionprops


class PipelineCombiner:
    """Intelligent combination of results from Pipeline A and Pipeline B"""
    
    def __init__(self, pipeline_a_results: Dict[str, torch.Tensor], 
                 pipeline_b_results: Dict[str, torch.Tensor]):
        self.pipeline_a_results = pipeline_a_results
        self.pipeline_b_results = pipeline_b_results
        
    def combine_by_region_performance(self, region_weights: Dict[str, float] = None) -> Dict[str, torch.Tensor]:
        """
        Combine results based on which pipeline performs better for each region type.
        
        Args:
            region_weights: Weights for different regions (WT, TC, ET)
        """
        if region_weights is None:
            region_weights = {
                'WT': 0.4,  # Whole Tumor - Pipeline A typically better
                'TC': 0.3,  # Tumor Core - Pipeline B typically better  
                'ET': 0.3   # Enhancing Tumor - Balanced
            }
        
        combined_results = {}
        
        for patient_id in self.pipeline_a_results.keys():
            if patient_id not in self.pipeline_b_results:
                print(f"Warning: No Pipeline B result for patient {patient_id}")
                combined_results[patient_id] = self.pipeline_a_results[patient_id]
                continue
                
            pred_a = self.pipeline_a_results[patient_id]
            pred_b = self.pipeline_b_results[patient_id]
            
            # Combine based on region-specific performance
            combined = self._combine_by_regions(pred_a, pred_b, region_weights)
            combined_results[patient_id] = combined
            
        return combined_results
    
    def _combine_by_regions(self, pred_a: torch.Tensor, pred_b: torch.Tensor, 
                          region_weights: Dict[str, float]) -> torch.Tensor:
        """Combine predictions based on region-specific performance"""
        # Convert to numpy for easier processing
        pred_a_np = pred_a.squeeze().cpu().numpy()
        pred_b_np = pred_b.squeeze().cpu().numpy()
        
        # Create region masks
        # WT: any tumor (labels 1, 2, 4)
        wt_mask_a = (pred_a_np > 0.5).astype(np.uint8)
        wt_mask_b = (pred_b_np > 0.5).astype(np.uint8)
        
        # TC: tumor core (labels 1, 4)
        tc_mask_a = ((pred_a_np == 1) | (pred_a_np == 4)).astype(np.uint8)
        tc_mask_b = ((pred_b_np == 1) | (pred_b_np == 4)).astype(np.uint8)
        
        # ET: enhancing tumor (label 4)
        et_mask_a = (pred_a_np == 4).astype(np.uint8)
        et_mask_b = (pred_b_np == 4).astype(np.uint8)
        
        # Combine based on region weights
        combined = np.zeros_like(pred_a_np)
        
        # For each region, choose the better performing pipeline
        for region, weight in region_weights.items():
            if region == 'WT':
                # Use Pipeline A for WT (typically better)
                combined = np.where(wt_mask_a, pred_a_np, combined)
            elif region == 'TC':
                # Use Pipeline B for TC (typically better)
                combined = np.where(tc_mask_b, pred_b_np, combined)
            elif region == 'ET':
                # Weighted combination for ET
                et_combined = weight * pred_a_np + (1 - weight) * pred_b_np
                combined = np.where(et_mask_a | et_mask_b, et_combined, combined)
        
        return torch.from_numpy(combined).unsqueeze(0).unsqueeze(0)
    
    def combine_by_confidence(self, confidence_threshold: float = 0.7) -> Dict[str, torch.Tensor]:
        """
        Combine results based on prediction confidence.
        Use the prediction with higher confidence for each voxel.
        """
        combined_results = {}
        
        for patient_id in self.pipeline_a_results.keys():
            if patient_id not in self.pipeline_b_results:
                combined_results[patient_id] = self.pipeline_a_results[patient_id]
                continue
                
            pred_a = self.pipeline_a_results[patient_id]
            pred_b = self.pipeline_b_results[patient_id]
            
            # Calculate confidence (max probability)
            conf_a = torch.max(pred_a, dim=1, keepdim=True)[0]
            conf_b = torch.max(pred_b, dim=1, keepdim=True)[0]
            
            # Choose prediction with higher confidence
            combined = torch.where(conf_a > conf_b, pred_a, pred_b)
            combined_results[patient_id] = combined
            
        return combined_results
    
    def combine_by_ensemble_averaging(self, weights: Tuple[float, float] = (0.5, 0.5)) -> Dict[str, torch.Tensor]:
        """
        Simple ensemble averaging with weights.
        
        Args:
            weights: (weight_a, weight_b) for Pipeline A and B
        """
        weight_a, weight_b = weights
        combined_results = {}
        
        for patient_id in self.pipeline_a_results.keys():
            if patient_id not in self.pipeline_b_results:
                combined_results[patient_id] = self.pipeline_a_results[patient_id]
                continue
                
            pred_a = self.pipeline_a_results[patient_id]
            pred_b = self.pipeline_b_results[patient_id]
            
            # Weighted average
            combined = weight_a * pred_a + weight_b * pred_b
            combined_results[patient_id] = combined
            
        return combined_results
    
    def combine_by_adaptive_weighting(self, validation_scores: Dict[str, Dict[str, float]] = None) -> Dict[str, torch.Tensor]:
        """
        Adaptive weighting based on validation performance.
        
        Args:
            validation_scores: Dict with patient_id -> {'pipeline_a': score, 'pipeline_b': score}
        """
        combined_results = {}
        
        for patient_id in self.pipeline_a_results.keys():
            if patient_id not in self.pipeline_b_results:
                combined_results[patient_id] = self.pipeline_a_results[patient_id]
                continue
                
            pred_a = self.pipeline_a_results[patient_id]
            pred_b = self.pipeline_b_results[patient_id]
            
            if validation_scores and patient_id in validation_scores:
                # Use validation scores to determine weights
                scores = validation_scores[patient_id]
                score_a = scores.get('pipeline_a', 0.5)
                score_b = scores.get('pipeline_b', 0.5)
                
                # Normalize scores to get weights
                total_score = score_a + score_b
                if total_score > 0:
                    weight_a = score_a / total_score
                    weight_b = score_b / total_score
                else:
                    weight_a = weight_b = 0.5
            else:
                # Default equal weighting
                weight_a = weight_b = 0.5
            
            # Weighted combination
            combined = weight_a * pred_a + weight_b * pred_b
            combined_results[patient_id] = combined
            
        return combined_results


class PostProcessor:
    """Post-processing for final segmentation results"""
    
    def __init__(self, min_tumor_size: int = 100, max_tumor_size: int = 1000000):
        self.min_tumor_size = min_tumor_size
        self.max_tumor_size = max_tumor_size
    
    def remove_small_components(self, segmentation: np.ndarray) -> np.ndarray:
        """Remove small connected components"""
        # Label connected components
        labeled = label(segmentation > 0)
        
        # Get properties of each component
        props = regionprops(labeled)
        
        # Create mask for components to keep
        keep_mask = np.zeros_like(segmentation, dtype=bool)
        
        for prop in props:
            if self.min_tumor_size <= prop.area <= self.max_tumor_size:
                keep_mask[labeled == prop.label] = True
        
        # Apply mask
        filtered_segmentation = np.where(keep_mask, segmentation, 0)
        
        return filtered_segmentation
    
    def fill_holes(self, segmentation: np.ndarray) -> np.ndarray:
        """Fill holes in segmentation"""
        from scipy.ndimage import binary_fill_holes
        
        # Fill holes for each class separately
        filled_segmentation = segmentation.copy()
        
        for class_id in [1, 2, 4]:  # Different tumor classes
            class_mask = (segmentation == class_id)
            if np.any(class_mask):
                filled_mask = binary_fill_holes(class_mask)
                filled_segmentation = np.where(filled_mask, class_id, filled_segmentation)
        
        return filled_segmentation
    
    def smooth_boundaries(self, segmentation: np.ndarray, sigma: float = 1.0) -> np.ndarray:
        """Smooth segmentation boundaries"""
        from scipy.ndimage import gaussian_filter
        
        # Apply Gaussian filter to each class
        smoothed_segmentation = segmentation.copy()
        
        for class_id in [1, 2, 4]:
            class_mask = (segmentation == class_id)
            if np.any(class_mask):
                smoothed_mask = gaussian_filter(class_mask.astype(float), sigma=sigma)
                # Threshold to get binary mask
                smoothed_mask = (smoothed_mask > 0.5).astype(int)
                smoothed_segmentation = np.where(smoothed_mask, class_id, smoothed_segmentation)
        
        return smoothed_segmentation
    
    def process_segmentation(self, segmentation: np.ndarray, 
                           remove_small: bool = True,
                           fill_holes: bool = True, 
                           smooth: bool = True) -> np.ndarray:
        """Apply all post-processing steps"""
        processed = segmentation.copy()
        
        if remove_small:
            processed = self.remove_small_components(processed)
        
        if fill_holes:
            processed = self.fill_holes(processed)
        
        if smooth:
            processed = self.smooth_boundaries(processed)
        
        return processed


def save_combined_results(combined_results: Dict[str, torch.Tensor], 
                         output_dir: Path,
                         post_process: bool = True) -> None:
    """Save combined results to files"""
    output_dir.mkdir(parents=True, exist_ok=True)
    post_processor = PostProcessor()
    
    for patient_id, segmentation in combined_results.items():
        # Convert to numpy
        seg_np = segmentation.squeeze().cpu().numpy()
        
        # Post-process if requested
        if post_process:
            seg_np = post_processor.process_segmentation(seg_np)
        
        # Convert to appropriate format (assuming BraTS format)
        # 0: background, 1: necrotic core, 2: edema, 4: enhancing tumor
        seg_np = seg_np.astype(np.uint8)
        
        # Save as NIfTI file
        output_path = output_dir / f"{patient_id}_seg.nii.gz"
        
        # Create NIfTI image (you may need to adjust header based on your data)
        nii_img = nib.Nifti1Image(seg_np, affine=np.eye(4))
        nib.save(nii_img, output_path)
        
        print(f"Saved segmentation for {patient_id} to {output_path}")


def evaluate_combination_strategies(pipeline_a_results: Dict[str, torch.Tensor],
                                 pipeline_b_results: Dict[str, torch.Tensor],
                                 ground_truth: Dict[str, torch.Tensor]) -> Dict[str, float]:
    """Evaluate different combination strategies"""
    from src.loss.dice import dice_score
    
    strategies = {
        'pipeline_a_only': pipeline_a_results,
        'pipeline_b_only': pipeline_b_results,
        'equal_weight': PipelineCombiner(pipeline_a_results, pipeline_b_results).combine_by_ensemble_averaging((0.5, 0.5)),
        'region_based': PipelineCombiner(pipeline_a_results, pipeline_b_results).combine_by_region_performance(),
        'confidence_based': PipelineCombiner(pipeline_a_results, pipeline_b_results).combine_by_confidence()
    }
    
    results = {}
    
    for strategy_name, predictions in strategies.items():
        total_dice = 0
        count = 0
        
        for patient_id in predictions.keys():
            if patient_id in ground_truth:
                pred = predictions[patient_id]
                gt = ground_truth[patient_id]
                
                # Calculate Dice score
                dice = dice_score(pred, gt)
                total_dice += dice
                count += 1
        
        if count > 0:
            results[strategy_name] = total_dice / count
        else:
            results[strategy_name] = 0.0
    
    return results
