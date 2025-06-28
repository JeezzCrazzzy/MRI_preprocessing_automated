#!/usr/bin/env python3
"""
Test script to compare different skull stripping methods
=======================================================

This script tests various skull stripping methods and creates
visualizations to help identify the best approach.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from mri_preprocessing_pipeline import MRIPreprocessingPipeline

def test_skull_stripping_methods():
    """
    Test different skull stripping methods on a sample image.
    """
    print("Testing skull stripping methods...")
    
    # Initialize pipeline
    pipeline = MRIPreprocessingPipeline()
    
    # Get the first available image
    input_files = [f for f in os.listdir("IXI-T1") if f.endswith('.nii.gz')]
    if not input_files:
        print("No input files found!")
        return
    
    input_file = os.path.join("IXI-T1", input_files[0])
    print(f"Testing with: {input_file}")
    
    # Load image
    img = pipeline.load_image(input_file)
    img_data = img.get_fdata()
    
    # Test different methods (start with the most reliable ones)
    methods = ['robust', 'otsu']  # Removed 'watershed' as it might have import issues
    
    # Check if watershed is available
    try:
        from skimage.feature import peak_local_maxima
        methods.append('watershed')
    except ImportError:
        try:
            from skimage.feature.peak import peak_local_maxima
            methods.append('watershed')
        except ImportError:
            print("Note: Watershed method not available due to import issues")
    
    results = {}
    
    for method in methods:
        print(f"\nTesting method: {method}")
        try:
            stripped, brain_mask = pipeline.skull_stripping(img_data, method=method)
            results[method] = {
                'stripped': stripped,
                'mask': brain_mask,
                'success': True
            }
            print(f"✓ {method} completed successfully")
        except Exception as e:
            print(f"✗ {method} failed: {str(e)}")
            results[method] = {
                'success': False,
                'error': str(e)
            }
    
    # Create comparison visualization
    create_comparison_visualization(img_data, results, input_files[0].replace('.nii.gz', ''))
    
    return results

def create_comparison_visualization(original, results, filename):
    """
    Create a visualization comparing different skull stripping methods.
    
    Parameters:
    -----------
    original : numpy.ndarray
        Original image data
    results : dict
        Results from different skull stripping methods
    filename : str
        Base filename for saving
    """
    print("Creating comparison visualization...")
    
    # Get middle slice for visualization
    mid_slice = original.shape[2] // 2
    
    # Calculate number of methods that succeeded
    successful_methods = [method for method, result in results.items() if result['success']]
    n_methods = len(successful_methods)
    
    if n_methods == 0:
        print("No successful methods to visualize")
        return
    
    # Create subplot grid
    fig, axes = plt.subplots(2, n_methods + 1, figsize=(5 * (n_methods + 1), 10))
    
    # Original image
    axes[0, 0].imshow(original[:, :, mid_slice], cmap='gray')
    axes[0, 0].set_title('Original Image')
    axes[0, 0].axis('off')
    
    axes[1, 0].imshow(original[:, :, mid_slice], cmap='gray')
    axes[1, 0].set_title('Original (zoomed)')
    axes[1, 0].axis('off')
    
    # Method results
    for i, method in enumerate(successful_methods):
        result = results[method]
        
        # Skull stripped image
        axes[0, i + 1].imshow(result['stripped'][:, :, mid_slice], cmap='gray')
        axes[0, i + 1].set_title(f'{method.upper()} - Stripped')
        axes[0, i + 1].axis('off')
        
        # Brain mask overlay
        mask_slice = result['mask'][:, :, mid_slice]
        original_slice = original[:, :, mid_slice]
        
        # Create overlay
        overlay = original_slice.copy()
        overlay[mask_slice == 0] = 0  # Set non-brain regions to black
        
        axes[1, i + 1].imshow(overlay, cmap='gray')
        axes[1, i + 1].set_title(f'{method.upper()} - Mask Overlay')
        axes[1, i + 1].axis('off')
    
    plt.tight_layout()
    plt.savefig(f'skull_stripping_comparison_{filename}.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Comparison saved as: skull_stripping_comparison_{filename}.png")

def test_center_mask_effect():
    """
    Test the effect of using brain-centered mask.
    """
    print("\nTesting brain-centered mask effect...")
    
    # Initialize pipeline
    pipeline = MRIPreprocessingPipeline()
    
    # Get the first available image
    input_files = [f for f in os.listdir("IXI-T1") if f.endswith('.nii.gz')]
    if not input_files:
        print("No input files found!")
        return
    
    input_file = os.path.join("IXI-T1", input_files[0])
    
    # Load image
    img = pipeline.load_image(input_file)
    img_data = img.get_fdata()
    
    # Test with and without center mask
    print("Testing robust method with center mask...")
    stripped_with_center, mask_with_center = pipeline.skull_stripping(
        img_data, method='robust', use_center_mask=True
    )
    
    print("Testing robust method without center mask...")
    stripped_without_center, mask_without_center = pipeline.skull_stripping(
        img_data, method='robust', use_center_mask=False
    )
    
    # Create comparison
    mid_slice = img_data.shape[2] // 2
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Original
    axes[0, 0].imshow(img_data[:, :, mid_slice], cmap='gray')
    axes[0, 0].set_title('Original')
    axes[0, 0].axis('off')
    
    # With center mask
    axes[0, 1].imshow(stripped_with_center[:, :, mid_slice], cmap='gray')
    axes[0, 1].set_title('With Center Mask')
    axes[0, 1].axis('off')
    
    # Without center mask
    axes[0, 2].imshow(stripped_without_center[:, :, mid_slice], cmap='gray')
    axes[0, 2].set_title('Without Center Mask')
    axes[0, 2].axis('off')
    
    # Mask comparisons
    axes[1, 0].imshow(mask_with_center[:, :, mid_slice], cmap='gray')
    axes[1, 0].set_title('Center Mask Applied')
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(mask_without_center[:, :, mid_slice], cmap='gray')
    axes[1, 1].set_title('No Center Mask')
    axes[1, 1].axis('off')
    
    axes[1, 2].axis('off')  # No difference visualization
    
    plt.tight_layout()
    plt.savefig('center_mask_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Center mask comparison saved as: center_mask_comparison.png")

def main():
    """
    Run all tests.
    """
    print("MRI Skull Stripping Test Suite")
    print("=" * 40)
    
    # Check if input directory exists
    if not os.path.exists("IXI-T1"):
        print("Error: IXI-T1 directory not found!")
        return
    
    try:
        # Test different methods
        results = test_skull_stripping_methods()
        
        # Test center mask effect
        test_center_mask_effect()
        
        print("\n" + "=" * 40)
        print("All tests completed!")
        print("Check the generated PNG files for visual comparisons.")
        
    except Exception as e:
        print(f"Error running tests: {str(e)}")

if __name__ == "__main__":
    main() 