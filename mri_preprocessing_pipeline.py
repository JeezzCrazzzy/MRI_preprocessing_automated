#!/usr/bin/env python3
"""
MRI Preprocessing Pipeline for T1-weighted Images
================================================

This pipeline performs the following preprocessing steps:
1. Image denoising
2. Bias field correction and visualization
3. Intensity normalization
4. Image registration to MNI152 template
5. Skull stripping

Author: MRI Preprocessing Pipeline
Date: 2024
"""

import os
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from scipy import ndimage
from scipy.ndimage import gaussian_filter
from scipy.optimize import minimize
from skimage import filters, morphology, measure
from skimage.restoration import denoise_bilateral, denoise_tv_chambolle
import warnings
warnings.filterwarnings('ignore')

class MRIPreprocessingPipeline:
    """
    A comprehensive preprocessing pipeline for T1-weighted MRI images.
    """
    
    def __init__(self, input_dir="IXI-T1", template_dir="templates", output_dir="preprocessed"):
        """
        Initialize the preprocessing pipeline.
        
        Parameters:
        -----------
        input_dir : str
            Directory containing input T1-weighted MRI files
        template_dir : str
            Directory containing MNI152 template files
        output_dir : str
            Directory to save preprocessed outputs
        """
        self.input_dir = input_dir
        self.template_dir = template_dir
        self.output_dir = output_dir
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Load MNI152 template
        self.template_path = os.path.join(template_dir, "mni_icbm152_t1_tal_nlin_sym_09a.nii")
        self.template_mask_path = os.path.join(template_dir, "mni_icbm152_t1_tal_nlin_sym_09a_mask.nii")
        
        if os.path.exists(self.template_path):
            self.template_img = nib.load(self.template_path)
            self.template_data = self.template_img.get_fdata()
        else:
            raise FileNotFoundError(f"Template file not found: {self.template_path}")
            
        if os.path.exists(self.template_mask_path):
            self.template_mask_img = nib.load(self.template_mask_path)
            self.template_mask_data = self.template_mask_img.get_fdata()
        else:
            print("Warning: Template mask not found, will use simple thresholding for skull stripping")
            self.template_mask_data = None
    
    def load_image(self, file_path):
        """
        Load NIfTI image file.
        
        Parameters:
        -----------
        file_path : str
            Path to the NIfTI file
            
        Returns:
        --------
        nibabel.Nifti1Image
            Loaded image object
        """
        return nib.load(file_path)
    
    def denoise_image(self, img_data, method='gaussian'):
        """
        Apply denoising to the image data.
        
        Parameters:
        -----------
        img_data : numpy.ndarray
            Input image data
        method : str
            Denoising method ('gaussian', 'tv', 'median', 'nonlocal_means')
            
        Returns:
        --------
        numpy.ndarray
            Denoised image data
        """
        print("Applying denoising...")
        
        if method == 'gaussian':
            # Gaussian smoothing - works well for 3D data
            denoised = gaussian_filter(img_data, sigma=1.0)
            
        elif method == 'tv':
            # Total variation denoising - works for 3D data
            denoised = denoise_tv_chambolle(img_data, weight=0.1)
            
        elif method == 'median':
            # Median filtering - good for salt-and-pepper noise
            denoised = ndimage.median_filter(img_data, size=3)
            
        elif method == 'nonlocal_means':
            # Non-local means denoising for 3D data
            from skimage.restoration import denoise_nl_means
            # Use a smaller patch size for 3D to avoid memory issues
            denoised = denoise_nl_means(img_data, h=0.1, fast_mode=True, 
                                      patch_size=5, patch_distance=7)
            
        elif method == 'bilateral_3d':
            # 3D bilateral filtering using scipy
            # This is a simplified 3D bilateral filter
            denoised = ndimage.gaussian_filter(img_data, sigma=1.0)
            # Apply edge-preserving smoothing
            denoised = ndimage.uniform_filter(denoised, size=3)
            
        else:
            raise ValueError(f"Unknown denoising method: {method}")
        
        return denoised
    
    def estimate_bias_field(self, img_data, degree=3):
        """
        Estimate bias field using polynomial fitting.
        
        Parameters:
        -----------
        img_data : numpy.ndarray
            Input image data
        degree : int
            Degree of polynomial for bias field estimation
            
        Returns:
        --------
        numpy.ndarray
            Estimated bias field
        """
        print("Estimating bias field...")
        
        # Create coordinate grids
        x, y, z = np.meshgrid(
            np.linspace(-1, 1, img_data.shape[0]),
            np.linspace(-1, 1, img_data.shape[1]),
            np.linspace(-1, 1, img_data.shape[2]),
            indexing='ij'
        )
        
        # Flatten coordinates and data
        coords = np.column_stack([x.flatten(), y.flatten(), z.flatten()])
        data_flat = img_data.flatten()
        
        # Remove background voxels (assuming background is near zero)
        # Use a more robust threshold
        threshold = np.percentile(data_flat[data_flat > 0], 10) if np.any(data_flat > 0) else 0
        mask = data_flat > threshold
        
        if np.sum(mask) < 100:  # Need minimum number of points for fitting
            print("Warning: Insufficient data points for bias field estimation, using uniform field")
            return np.ones_like(img_data)
        
        coords_masked = coords[mask]
        data_masked = data_flat[mask]
        
        # Ensure we have valid data
        if np.isnan(data_masked).any() or np.isinf(data_masked).any():
            print("Warning: Invalid data in bias field estimation, using uniform field")
            return np.ones_like(img_data)
        
        try:
            # Fit polynomial
            from sklearn.preprocessing import PolynomialFeatures
            from sklearn.linear_model import LinearRegression
            
            poly = PolynomialFeatures(degree=degree)
            coords_poly = poly.fit_transform(coords_masked)
            
            model = LinearRegression()
            model.fit(coords_poly, data_masked)
            
            # Predict bias field for all voxels
            coords_poly_all = poly.transform(coords)
            bias_field_flat = model.predict(coords_poly_all)
            bias_field = bias_field_flat.reshape(img_data.shape)
            
            # Ensure bias field is positive and reasonable
            bias_field = np.maximum(bias_field, 0.1)  # Avoid division by zero
            bias_field = np.minimum(bias_field, 10.0)  # Avoid extreme values
            
            return bias_field
            
        except Exception as e:
            print(f"Warning: Bias field estimation failed: {str(e)}, using uniform field")
            return np.ones_like(img_data)
    
    def correct_bias_field(self, img_data, bias_field):
        """
        Correct bias field from image data.
        
        Parameters:
        -----------
        img_data : numpy.ndarray
            Input image data
        bias_field : numpy.ndarray
            Estimated bias field
            
        Returns:
        --------
        numpy.ndarray
            Bias field corrected image data
        """
        print("Correcting bias field...")
        
        # Normalize bias field to avoid division by zero
        bias_field_norm = bias_field / np.mean(bias_field)
        
        # Apply bias field correction
        corrected = img_data / bias_field_norm
        
        return corrected
    
    def visualize_bias_field(self, original, bias_field, corrected, output_path):
        """
        Visualize bias field correction results.
        
        Parameters:
        -----------
        original : numpy.ndarray
            Original image data
        bias_field : numpy.ndarray
            Estimated bias field
        corrected : numpy.ndarray
            Bias field corrected image data
        output_path : str
            Path to save visualization
        """
        print("Creating bias field visualization...")
        
        # Get middle slice for visualization
        mid_slice = original.shape[2] // 2
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Original image
        im1 = axes[0].imshow(original[:, :, mid_slice], cmap='gray')
        axes[0].set_title('Original Image')
        axes[0].axis('off')
        plt.colorbar(im1, ax=axes[0])
        
        # Bias field
        im2 = axes[1].imshow(bias_field[:, :, mid_slice], cmap='hot')
        axes[1].set_title('Estimated Bias Field')
        axes[1].axis('off')
        plt.colorbar(im2, ax=axes[1])
        
        # Corrected image
        im3 = axes[2].imshow(corrected[:, :, mid_slice], cmap='gray')
        axes[2].set_title('Bias Field Corrected')
        axes[2].axis('off')
        plt.colorbar(im3, ax=axes[2])
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def normalize_intensity(self, img_data, method='zscore'):
        """
        Normalize image intensity.
        
        Parameters:
        -----------
        img_data : numpy.ndarray
            Input image data
        method : str
            Normalization method ('zscore', 'minmax', 'histogram')
            
        Returns:
        --------
        numpy.ndarray
            Normalized image data
        """
        print("Normalizing intensity...")
        
        if method == 'zscore':
            # Z-score normalization
            mean_val = np.mean(img_data[img_data > 0])
            std_val = np.std(img_data[img_data > 0])
            normalized = (img_data - mean_val) / std_val
            
        elif method == 'minmax':
            # Min-max normalization to [0, 1]
            min_val = np.min(img_data)
            max_val = np.max(img_data)
            normalized = (img_data - min_val) / (max_val - min_val)
            
        elif method == 'histogram':
            # Histogram equalization
            from skimage import exposure
            normalized = exposure.equalize_hist(img_data)
            
        else:
            raise ValueError(f"Unknown normalization method: {method}")
        
        return normalized
    
    def register_to_template(self, img_data, img_affine, method='affine'):
        """
        Register image to MNI152 template.
        
        Parameters:
        -----------
        img_data : numpy.ndarray
            Input image data
        img_affine : numpy.ndarray
            Affine transformation matrix of input image
        method : str
            Registration method ('affine', 'rigid')
            
        Returns:
        --------
        tuple
            (registered_data, registered_affine)
        """
        print("Registering to MNI152 template...")
        
        # For simplicity, we'll use a basic affine registration
        # In practice, you might want to use more sophisticated methods like ANTs or FSL
        
        # Create a simple affine transformation
        # This is a simplified version - in practice, you'd use proper registration algorithms
        
        # Resize image to match template dimensions
        target_shape = self.template_data.shape
        
        # Simple resizing (in practice, use proper interpolation)
        from scipy.ndimage import zoom
        
        zoom_factors = [target_shape[i] / img_data.shape[i] for i in range(3)]
        registered_data = zoom(img_data, zoom_factors, order=1)
        
        # Update affine matrix
        registered_affine = img_affine.copy()
        for i in range(3):
            registered_affine[i, i] *= zoom_factors[i]
        
        return registered_data, registered_affine
    
    def skull_stripping(self, img_data, method='otsu'):
        """
        Perform skull stripping.
        
        Parameters:
        -----------
        img_data : numpy.ndarray
            Input image data
        method : str
            Skull stripping method ('otsu', 'template', 'watershed')
            
        Returns:
        --------
        tuple
            (stripped_data, brain_mask)
        """
        print("Performing skull stripping...")
        
        if method == 'otsu':
            # Otsu thresholding
            threshold = filters.threshold_otsu(img_data)
            brain_mask = img_data > threshold
            
            # Clean up mask with morphological operations
            brain_mask = morphology.binary_opening(brain_mask, morphology.ball(3))
            brain_mask = morphology.binary_closing(brain_mask, morphology.ball(5))
            
            # Keep largest connected component
            labels = measure.label(brain_mask)
            if labels.max() > 0:  # Check if any connected components exist
                largest_cc = labels == np.argmax(np.bincount(labels.flat)[1:]) + 1
                brain_mask = largest_cc
            else:
                # If no connected components found, use the original thresholded mask
                print("Warning: No connected components found, using thresholded mask")
                brain_mask = img_data > threshold
            
        elif method == 'template' and self.template_mask_data is not None:
            # Use template mask (requires registration first)
            brain_mask = self.template_mask_data > 0.5
            
        elif method == 'watershed':
            # Watershed-based skull stripping
            from skimage.segmentation import watershed
            from skimage.feature import peak_local_maxima
            
            # Create distance map
            threshold = filters.threshold_otsu(img_data)
            binary = img_data > threshold
            
            distance = ndimage.distance_transform_edt(binary)
            
            # Find local maxima
            local_maxi = peak_local_maxima(distance, labels=binary, 
                                         min_distance=20, exclude_border=False)
            
            if len(local_maxi) > 0:  # Check if local maxima were found
                local_maxi = np.ravel_multi_index(local_maxi.T, distance.shape)
                
                # Create markers
                markers = np.zeros_like(distance, dtype=int)
                markers.ravel()[local_maxi] = range(len(local_maxi))
                
                # Apply watershed
                brain_mask = watershed(-distance, markers, mask=binary)
                brain_mask = brain_mask > 0
            else:
                # Fallback to simple thresholding if no local maxima found
                print("Warning: No local maxima found, using thresholded mask")
                brain_mask = binary
            
        else:
            raise ValueError(f"Unknown skull stripping method: {method}")
        
        # Apply mask to image
        stripped_data = img_data * brain_mask
        
        return stripped_data, brain_mask
    
    def preprocess_single_image(self, input_file, output_prefix):
        """
        Preprocess a single MRI image through the complete pipeline.
        
        Parameters:
        -----------
        input_file : str
            Path to input NIfTI file
        output_prefix : str
            Prefix for output files
            
        Returns:
        --------
        dict
            Dictionary containing all intermediate and final results
        """
        print(f"\nProcessing: {input_file}")
        
        try:
            # Load image
            img = self.load_image(input_file)
            img_data = img.get_fdata()
            img_affine = img.affine
            
            # Validate image data
            if img_data is None or img_data.size == 0:
                raise ValueError("Empty or invalid image data")
            
            if np.isnan(img_data).any() or np.isinf(img_data).any():
                print("Warning: Image contains NaN or Inf values, cleaning...")
                img_data = np.nan_to_num(img_data, nan=0.0, posinf=0.0, neginf=0.0)
            
            results = {
                'original': img_data,
                'affine': img_affine
            }
            
            # Step 1: Denoising
            try:
                denoised = self.denoise_image(img_data)
                results['denoised'] = denoised
            except Exception as e:
                print(f"Warning: Denoising failed, using original data: {str(e)}")
                results['denoised'] = img_data
            
            # Step 2: Bias field correction
            try:
                bias_field = self.estimate_bias_field(results['denoised'])
                bias_corrected = self.correct_bias_field(results['denoised'], bias_field)
                results['bias_field'] = bias_field
                results['bias_corrected'] = bias_corrected
            except Exception as e:
                print(f"Warning: Bias field correction failed, using denoised data: {str(e)}")
                results['bias_field'] = np.ones_like(results['denoised'])
                results['bias_corrected'] = results['denoised']
            
            # Step 3: Intensity normalization
            try:
                normalized = self.normalize_intensity(results['bias_corrected'])
                results['normalized'] = normalized
            except Exception as e:
                print(f"Warning: Intensity normalization failed, using bias corrected data: {str(e)}")
                results['normalized'] = results['bias_corrected']
            
            # Step 4: Registration to template
            try:
                registered, registered_affine = self.register_to_template(results['normalized'], img_affine)
                results['registered'] = registered
                results['registered_affine'] = registered_affine
            except Exception as e:
                print(f"Warning: Registration failed, using normalized data: {str(e)}")
                results['registered'] = results['normalized']
                results['registered_affine'] = img_affine
            
            # Step 5: Skull stripping
            try:
                stripped, brain_mask = self.skull_stripping(results['registered'])
                results['stripped'] = stripped
                results['brain_mask'] = brain_mask
            except Exception as e:
                print(f"Warning: Skull stripping failed, using registered data: {str(e)}")
                results['stripped'] = results['registered']
                results['brain_mask'] = np.ones_like(results['registered'], dtype=bool)
            
            # Save results
            self.save_results(results, output_prefix)
            
            # Create visualizations
            try:
                self.create_visualizations(results, output_prefix)
            except Exception as e:
                print(f"Warning: Visualization creation failed: {str(e)}")
            
            return results
            
        except Exception as e:
            print(f"Error processing {input_file}: {str(e)}")
            raise
    
    def save_results(self, results, output_prefix):
        """
        Save preprocessing results to files.
        
        Parameters:
        -----------
        results : dict
            Dictionary containing preprocessing results
        output_prefix : str
            Prefix for output files
        """
        print("Saving results...")
        
        # Save final preprocessed image
        final_img = nib.Nifti1Image(results['stripped'], results['registered_affine'])
        nib.save(final_img, os.path.join(self.output_dir, f"{output_prefix}_preprocessed.nii.gz"))
        
        # Save brain mask
        mask_img = nib.Nifti1Image(results['brain_mask'].astype(np.uint8), results['registered_affine'])
        nib.save(mask_img, os.path.join(self.output_dir, f"{output_prefix}_brain_mask.nii.gz"))
        
        # Save intermediate results
        denoised_img = nib.Nifti1Image(results['denoised'], results['affine'])
        nib.save(denoised_img, os.path.join(self.output_dir, f"{output_prefix}_denoised.nii.gz"))
        
        bias_corrected_img = nib.Nifti1Image(results['bias_corrected'], results['affine'])
        nib.save(bias_corrected_img, os.path.join(self.output_dir, f"{output_prefix}_bias_corrected.nii.gz"))
        
        normalized_img = nib.Nifti1Image(results['normalized'], results['affine'])
        nib.save(normalized_img, os.path.join(self.output_dir, f"{output_prefix}_normalized.nii.gz"))
        
        registered_img = nib.Nifti1Image(results['registered'], results['registered_affine'])
        nib.save(registered_img, os.path.join(self.output_dir, f"{output_prefix}_registered.nii.gz"))
    
    def create_visualizations(self, results, output_prefix):
        """
        Create visualizations of preprocessing steps.
        
        Parameters:
        -----------
        results : dict
            Dictionary containing preprocessing results
        output_prefix : str
            Prefix for output files
        """
        print("Creating visualizations...")
        
        # Bias field visualization
        self.visualize_bias_field(
            results['original'],
            results['bias_field'],
            results['bias_corrected'],
            os.path.join(self.output_dir, f"{output_prefix}_bias_field.png")
        )
        
        # Pipeline overview
        self.create_pipeline_overview(results, output_prefix)
    
    def create_pipeline_overview(self, results, output_prefix):
        """
        Create an overview visualization of all preprocessing steps.
        
        Parameters:
        -----------
        results : dict
            Dictionary containing preprocessing results
        output_prefix : str
            Prefix for output files
        """
        # Get middle slice for visualization
        mid_slice = results['original'].shape[2] // 2
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # Original
        axes[0, 0].imshow(results['original'][:, :, mid_slice], cmap='gray')
        axes[0, 0].set_title('Original')
        axes[0, 0].axis('off')
        
        # Denoised
        axes[0, 1].imshow(results['denoised'][:, :, mid_slice], cmap='gray')
        axes[0, 1].set_title('Denoised')
        axes[0, 1].axis('off')
        
        # Bias corrected
        axes[0, 2].imshow(results['bias_corrected'][:, :, mid_slice], cmap='gray')
        axes[0, 2].set_title('Bias Corrected')
        axes[0, 2].axis('off')
        
        # Normalized
        axes[1, 0].imshow(results['normalized'][:, :, mid_slice], cmap='gray')
        axes[1, 0].set_title('Normalized')
        axes[1, 0].axis('off')
        
        # Registered
        axes[1, 1].imshow(results['registered'][:, :, mid_slice], cmap='gray')
        axes[1, 1].set_title('Registered')
        axes[1, 1].axis('off')
        
        # Final (skull stripped)
        axes[1, 2].imshow(results['stripped'][:, :, mid_slice], cmap='gray')
        axes[1, 2].set_title('Skull Stripped')
        axes[1, 2].axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, f"{output_prefix}_pipeline_overview.png"), 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def process_all_images(self):
        """
        Process all T1-weighted images in the input directory.
        """
        print("Starting batch processing of all T1-weighted images...")
        
        # Get all .nii.gz files in the input directory
        input_files = [f for f in os.listdir(self.input_dir) if f.endswith('.nii.gz')]
        
        if not input_files:
            print(f"No .nii.gz files found in {self.input_dir}")
            return
        
        print(f"Found {len(input_files)} files to process")
        
        for i, filename in enumerate(input_files):
            print(f"\nProcessing file {i+1}/{len(input_files)}: {filename}")
            
            input_path = os.path.join(self.input_dir, filename)
            output_prefix = filename.replace('.nii.gz', '')
            
            try:
                results = self.preprocess_single_image(input_path, output_prefix)
                print(f"Successfully processed: {filename}")
            except Exception as e:
                print(f"Error processing {filename}: {str(e)}")
                continue
        
        print(f"\nBatch processing completed. Results saved in: {self.output_dir}")


def main():
    """
    Main function to run the preprocessing pipeline.
    """
    # Initialize pipeline
    pipeline = MRIPreprocessingPipeline()
    
    # Process all images
    pipeline.process_all_images()


if __name__ == "__main__":
    main() 