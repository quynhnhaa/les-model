"""Self-ensemble training module for creating pseudo-labels from multiple checkpoints"""
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple
import torch.nn.functional as F


class SelfEnsembleTrainer:
    """Self-ensemble trainer that creates pseudo-labels from multiple model checkpoints"""
    
    def __init__(self, model_class, model_args, device='cuda'):
        self.model_class = model_class
        self.model_args = model_args
        self.device = device
        self.models = []
        
    def load_checkpoints(self, checkpoint_paths: List[Path]) -> None:
        """Load multiple model checkpoints"""
        self.models = []
        
        for checkpoint_path in checkpoint_paths:
            if not checkpoint_path.exists():
                print(f"Warning: Checkpoint {checkpoint_path} not found")
                continue
                
            # Create model instance
            model = self.model_class(**self.model_args)
            
            # Load checkpoint
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            if 'state_dict' in checkpoint:
                model.load_state_dict(checkpoint['state_dict'])
            else:
                model.load_state_dict(checkpoint)
                
            model.eval()
            model.to(self.device)
            self.models.append(model)
            
        print(f"Loaded {len(self.models)} models for self-ensemble")
    
    def generate_pseudo_labels(self, dataloader, tta_transforms=None) -> Dict[str, torch.Tensor]:
        """Generate pseudo-labels by averaging predictions from multiple models"""
        pseudo_labels = {}
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloader):
                patient_id = batch["patient_id"][0]
                inputs = batch["image"].cuda()
                
                # Collect predictions from all models
                all_predictions = []
                
                for model in self.models:
                    if tta_transforms:
                        # Apply TTA
                        predictions = self._apply_tta(model, inputs, tta_transforms)
                    else:
                        # Single prediction
                        if hasattr(model, 'deep_supervision') and model.deep_supervision:
                            predictions, _ = model(inputs)
                        else:
                            predictions = model(inputs)
                        predictions = torch.sigmoid(predictions)
                    
                    all_predictions.append(predictions.cpu())
                
                # Average predictions to create pseudo-labels
                pseudo_label = torch.stack(all_predictions).mean(dim=0)
                pseudo_labels[patient_id] = pseudo_label
                
                if batch_idx % 10 == 0:
                    print(f"Generated pseudo-labels for {batch_idx + 1}/{len(dataloader)} batches")
        
        return pseudo_labels
    
    def _apply_tta(self, model, inputs, tta_transforms):
        """Apply Test Time Augmentation"""
        predictions = []
        
        for transform, inverse_transform in tta_transforms:
            # Apply transformation
            transformed_inputs = transform(inputs)
            
            # Get prediction
            if hasattr(model, 'deep_supervision') and model.deep_supervision:
                pred, _ = model(transformed_inputs)
            else:
                pred = model(transformed_inputs)
            
            # Apply inverse transformation
            pred = inverse_transform(pred)
            predictions.append(torch.sigmoid(pred))
        
        # Average predictions
        return torch.stack(predictions).mean(dim=0)
    
    def create_ensemble_dataset(self, original_dataset, pseudo_labels, alpha=0.5):
        """Create dataset that combines original labels with pseudo-labels"""
        class EnsembleDataset:
            def __init__(self, original_dataset, pseudo_labels, alpha):
                self.original_dataset = original_dataset
                self.pseudo_labels = pseudo_labels
                self.alpha = alpha
                
            def __len__(self):
                return len(self.original_dataset)
            
            def __getitem__(self, idx):
                # Get original sample
                sample = self.original_dataset[idx]
                patient_id = sample["patient_id"]
                
                if patient_id in self.pseudo_labels:
                    # Combine original and pseudo labels
                    original_label = sample["label"]
                    pseudo_label = self.pseudo_labels[patient_id]
                    
                    # Weighted combination
                    combined_label = self.alpha * original_label + (1 - self.alpha) * pseudo_label
                    sample["label"] = combined_label
                
                return sample
        
        return EnsembleDataset(original_dataset, pseudo_labels, alpha)


class PseudoLabelGenerator:
    """Generate pseudo-labels using multiple training strategies"""
    
    def __init__(self, model_class, model_args, device='cuda'):
        self.model_class = model_class
        self.model_args = model_args
        self.device = device
    
    def generate_from_different_seeds(self, checkpoint_dir: Path, seeds: List[int], 
                                    dataloader, tta_transforms=None) -> Dict[str, torch.Tensor]:
        """Generate pseudo-labels from models trained with different seeds"""
        ensemble_trainer = SelfEnsembleTrainer(self.model_class, self.model_args, self.device)
        
        # Find checkpoints for different seeds
        checkpoint_paths = []
        for seed in seeds:
            seed_checkpoints = list(checkpoint_dir.glob(f"*seed{seed}*/*.pth.tar"))
            if seed_checkpoints:
                checkpoint_paths.append(seed_checkpoints[0])  # Take the best checkpoint
        
        if not checkpoint_paths:
            raise ValueError(f"No checkpoints found in {checkpoint_dir}")
        
        ensemble_trainer.load_checkpoints(checkpoint_paths)
        return ensemble_trainer.generate_pseudo_labels(dataloader, tta_transforms)
    
    def generate_from_different_architectures(self, checkpoint_dirs: List[Path], 
                                           dataloader, tta_transforms=None) -> Dict[str, torch.Tensor]:
        """Generate pseudo-labels from models with different architectures"""
        ensemble_trainer = SelfEnsembleTrainer(self.model_class, self.model_args, self.device)
        
        # Find best checkpoints from different architecture directories
        checkpoint_paths = []
        for checkpoint_dir in checkpoint_dirs:
            best_checkpoints = list(checkpoint_dir.glob("model_best.pth.tar"))
            if best_checkpoints:
                checkpoint_paths.append(best_checkpoints[0])
        
        if not checkpoint_paths:
            raise ValueError(f"No checkpoints found in {checkpoint_dirs}")
        
        ensemble_trainer.load_checkpoints(checkpoint_paths)
        return ensemble_trainer.generate_pseudo_labels(dataloader, tta_transforms)


def create_advanced_tta_transforms():
    """Create advanced TTA transforms for self-ensemble"""
    from src.tta import simple_tta
    
    def advanced_tta(x):
        """Advanced TTA with more transformations"""
        # Get basic TTA transforms
        basic_transforms = simple_tta(x)
        
        # Add additional transformations
        advanced_transforms = []
        
        # Add intensity variations
        for intensity_factor in [0.8, 1.0, 1.2]:
            def intensity_transform(img, factor=intensity_factor):
                return img * factor
            
            def intensity_inverse(pred, factor=intensity_factor):
                return pred  # No inverse needed for intensity
            
            advanced_transforms.append((intensity_transform, intensity_inverse))
        
        # Add elastic deformation (simplified)
        for deformation_strength in [0.0, 0.1, 0.2]:
            def elastic_transform(img, strength=deformation_strength):
                if strength == 0.0:
                    return img
                # Simplified elastic deformation
                return img  # Placeholder for actual elastic deformation
            
            def elastic_inverse(pred, strength=deformation_strength):
                return pred  # Placeholder for inverse elastic deformation
            
            advanced_transforms.append((elastic_transform, elastic_inverse))
        
        return basic_transforms + advanced_transforms
    
    return advanced_tta
