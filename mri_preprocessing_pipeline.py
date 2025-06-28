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
    
    def estimate_bias_field(self, img_data, degree=3, method='conservative'):
        """
        Estimate bias field using different methods.
        
        Parameters:
        -----------
        img_data : numpy.ndarray
            Input image data
        degree : int
            Degree of polynomial for bias field estimation
        method : str
            Method to use ('polynomial', 'gaussian', 'conservative')
            
        Returns:
        --------
        numpy.ndarray
            Estimated bias field
        """
        print("Estimating bias field...")
        
        if method == 'conservative':
            # Use a very conservative approach with minimal correction
            return self._estimate_conservative_bias_field(img_data)
        elif method == 'gaussian':
            # Use Gaussian smoothing approach
            return self._estimate_gaussian_bias_field(img_data)
        else:
            # Use polynomial fitting (default)
            return self._estimate_polynomial_bias_field(img_data, degree)
    
    def _estimate_conservative_bias_field(self, img_data):
        """
        Estimate bias field using a very conservative approach.
        """
        # Use a simple Gaussian smoothing to estimate slow intensity variations
        bias_field = gaussian_filter(img_data, sigma=20)
        
        # Normalize to be close to 1.0
        bias_field_mean = np.mean(bias_field)
        bias_field = bias_field / bias_field_mean
        
        # Very conservative bounds
        bias_field = np.clip(bias_field, 0.8, 1.2)
        
        return bias_field
    
    def _estimate_gaussian_bias_field(self, img_data):
        """
        Estimate bias field using Gaussian smoothing.
        """
        # Apply heavy Gaussian smoothing to get the low-frequency component
        smoothed = gaussian_filter(img_data, sigma=15)
        
        # Normalize
        smoothed_mean = np.mean(smoothed)
        bias_field = smoothed / smoothed_mean
        
        # Conservative bounds
        bias_field = np.clip(bias_field, 0.7, 1.3)
        
        return bias_field
    
    def _estimate_polynomial_bias_field(self, img_data, degree=3):
        """
        Estimate bias field using polynomial fitting.
        """
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
            
            # Normalize bias field to be close to 1.0 to avoid extreme corrections
            bias_field_mean = np.mean(bias_field)
            bias_field = bias_field / bias_field_mean
            
            # Ensure bias field is within reasonable bounds
            bias_field = np.clip(bias_field, 0.5, 2.0)
            
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
        
        # Ensure bias field is positive and reasonable
        bias_field = np.maximum(bias_field, 0.1)  # Avoid division by zero
        
        # Apply bias field correction
        corrected = img_data / bias_field
        
        # Clip to prevent extreme values
        original_range = np.percentile(img_data[img_data > 0], [1, 99])
        corrected = np.clip(corrected, original_range[0], original_range[1])
        
        # Ensure we don't lose the original image structure
        if np.std(corrected) < np.std(img_data) * 0.1:
            print("Warning: Bias correction too aggressive, using original image")
            return img_data
        
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
    
    def skull_stripping(self, img_data, method='robust', use_center_mask=True):
        """
        Perform skull stripping.
        
        Parameters:
        -----------
        img_data : numpy.ndarray
            Input image data
        method : str
            Skull stripping method ('otsu', 'template', 'watershed', 'robust')
        use_center_mask : bool
            Whether to apply brain-centered mask for better results
            
        Returns:
        --------
        tuple
            (stripped_data, brain_mask)
        """
        print("Performing skull stripping...")
        
        if method == 'otsu':
            brain_mask = self._robust_skull_stripping(img_data)
            
        elif method == 'template' and self.template_mask_data is not None:
            # Use template mask (requires registration first)
            brain_mask = self.template_mask_data > 0.5
            
        elif method == 'watershed':
            # Watershed-based skull stripping
            brain_mask = self._watershed_skull_stripping(img_data)
            
        elif method == 'robust':
            # Most robust method combining multiple approaches
            brain_mask = self._robust_skull_stripping(img_data)
            
        else:
            raise ValueError(f"Unknown skull stripping method: {method}")
        
        # Optionally apply brain-centered mask for better results
        if use_center_mask and method != 'template':
            brain_mask = self._create_brain_centered_mask(img_data, brain_mask)
        
        # Apply mask to image
        stripped_data = img_data * brain_mask
        
        return stripped_data, brain_mask
    
    def _robust_skull_stripping(self, img_data):
        """
        Robust skull stripping using multiple thresholding and morphological operations.
        
        Parameters:
        -----------
        img_data : numpy.ndarray
            Input image data
            
        Returns:
        --------
        numpy.ndarray
            Binary brain mask
        """
        # Step 1: Initial thresholding using Otsu
        threshold = filters.threshold_otsu(img_data)
        initial_mask = img_data > threshold
        
        # Step 2: Remove small objects and fill holes
        initial_mask = morphology.remove_small_objects(initial_mask, min_size=1000)
        
        # Handle different scikit-image versions for binary_fill_holes
        try:
            initial_mask = morphology.binary_fill_holes(initial_mask)
        except AttributeError:
            try:
                from skimage.morphology import binary_fill_holes
                initial_mask = binary_fill_holes(initial_mask)
            except ImportError:
                # Fallback: use scipy's binary_fill_holes
                from scipy.ndimage import binary_fill_holes
                initial_mask = binary_fill_holes(initial_mask)
        
        # Step 3: Erode to remove skull and other non-brain tissue
        selem = morphology.ball(3)
        eroded_mask = morphology.binary_erosion(initial_mask, selem)
        
        # Step 4: Find connected components and keep the largest
        labels = measure.label(eroded_mask)
        if labels.max() > 0:
            # Get the largest connected component
            largest_cc = labels == np.argmax(np.bincount(labels.flat)[1:]) + 1
            brain_mask = largest_cc
        else:
            print("Warning: No connected components found after erosion, using original mask")
            brain_mask = initial_mask
        
        # Step 5: Dilate to restore brain boundaries
        selem = morphology.ball(2)
        brain_mask = morphology.binary_dilation(brain_mask, selem)
        
        # Step 6: Apply additional morphological operations for smoothness
        brain_mask = morphology.binary_closing(brain_mask, morphology.ball(3))
        brain_mask = morphology.binary_opening(brain_mask, morphology.ball(2))
        
        # Step 7: Fill any remaining holes
        try:
            brain_mask = morphology.binary_fill_holes(brain_mask)
        except AttributeError:
            try:
                from skimage.morphology import binary_fill_holes
                brain_mask = binary_fill_holes(brain_mask)
            except ImportError:
                # Fallback: use scipy's binary_fill_holes
                from scipy.ndimage import binary_fill_holes
                brain_mask = binary_fill_holes(brain_mask)
        
        # Step 8: Remove any remaining small objects
        brain_mask = morphology.remove_small_objects(brain_mask, min_size=500)
        
        return brain_mask
    
    def _watershed_skull_stripping(self, img_data):
        """
        Watershed-based skull stripping.
        
        Parameters:
        -----------
        img_data : numpy.ndarray
            Input image data
            
        Returns:
        --------
        numpy.ndarray
            Binary brain mask
        """
        from skimage.segmentation import watershed
        try:
            # Try the new location first (scikit-image >= 0.20)
            from skimage.feature import peak_local_maxima
        except ImportError:
            try:
                # Try the old location (scikit-image < 0.20)
                from skimage.feature.peak import peak_local_maxima
            except ImportError:
                try:
                    # Try the very old location
                    from skimage.feature import peak_local_maxima
                except ImportError:
                    # Fallback: use a simple approach without peak detection
                    print("Warning: peak_local_maxima not available, using fallback method")
                    return self._robust_skull_stripping(img_data)
        
        # Create distance map
        threshold = filters.threshold_otsu(img_data)
        binary = img_data > threshold
        
        # Clean up binary image
        binary = morphology.remove_small_objects(binary, min_size=1000)
        
        # Handle different scikit-image versions for binary_fill_holes
        try:
            binary = morphology.binary_fill_holes(binary)
        except AttributeError:
            try:
                from skimage.morphology import binary_fill_holes
                binary = binary_fill_holes(binary)
            except ImportError:
                # Fallback: use scipy's binary_fill_holes
                from scipy.ndimage import binary_fill_holes
                binary = binary_fill_holes(binary)
        
        distance = ndimage.distance_transform_edt(binary)
        
        # Find local maxima with more conservative parameters
        try:
            # Try different parameter combinations for different scikit-image versions
            try:
                local_maxi = peak_local_maxima(distance, labels=binary, 
                                             min_distance=15, exclude_border=False,
                                             threshold_abs=0.1)
            except TypeError:
                # Older version might not have threshold_abs parameter
                local_maxi = peak_local_maxima(distance, labels=binary, 
                                             min_distance=15, exclude_border=False)
            
            if len(local_maxi) > 0:
                local_maxi = np.ravel_multi_index(local_maxi.T, distance.shape)
                
                # Create markers
                markers = np.zeros_like(distance, dtype=int)
                markers.ravel()[local_maxi] = range(len(local_maxi))
                
                # Apply watershed
                brain_mask = watershed(-distance, markers, mask=binary)
                brain_mask = brain_mask > 0
                
                # Clean up watershed result
                brain_mask = morphology.remove_small_objects(brain_mask, min_size=1000)
                
                # Handle different scikit-image versions for binary_fill_holes
                try:
                    brain_mask = morphology.binary_fill_holes(brain_mask)
                except AttributeError:
                    try:
                        from skimage.morphology import binary_fill_holes
                        brain_mask = binary_fill_holes(brain_mask)
                    except ImportError:
                        # Fallback: use scipy's binary_fill_holes
                        from scipy.ndimage import binary_fill_holes
                        brain_mask = binary_fill_holes(brain_mask)
                
            else:
                print("Warning: No local maxima found, using fallback method")
                brain_mask = self._robust_skull_stripping(img_data)
                
        except Exception as e:
            print(f"Warning: Watershed failed: {str(e)}, using fallback method")
            brain_mask = self._robust_skull_stripping(img_data)
        
        return brain_mask
    
    def _create_brain_centered_mask(self, img_data, brain_mask):
        """
        Create a mask that focuses on the central brain region.
        
        Parameters:
        -----------
        img_data : numpy.ndarray
            Input image data
        brain_mask : numpy.ndarray
            Initial brain mask
            
        Returns:
        --------
        numpy.ndarray
            Refined brain mask
        """
        # Get the center of the brain
        brain_coords = np.where(brain_mask)
        if len(brain_coords[0]) == 0:
            return brain_mask
        
        center_x = np.mean(brain_coords[0])
        center_y = np.mean(brain_coords[1])
        center_z = np.mean(brain_coords[2])
        
        # Create a spherical mask around the brain center
        x, y, z = np.meshgrid(np.arange(img_data.shape[0]),
                             np.arange(img_data.shape[1]),
                             np.arange(img_data.shape[2]), indexing='ij')
        
        # Calculate distance from center
        radius = min(img_data.shape) * 0.4  # Adjust radius as needed
        distance = np.sqrt((x - center_x)**2 + (y - center_y)**2 + (z - center_z)**2)
        
        # Create spherical mask
        spherical_mask = distance <= radius
        
        # Combine with original brain mask
        refined_mask = brain_mask & spherical_mask
        
        return refined_mask
    
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
                bias_field = self.estimate_bias_field(results['denoised'], method='conservative')
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
        
        # Get affine matrices with fallbacks
        registered_affine = results.get('registered_affine', results.get('affine'))
        original_affine = results.get('affine')
        
        # Save final preprocessed image
        if 'stripped' in results:
            final_img = nib.Nifti1Image(results['stripped'], registered_affine)
            nib.save(final_img, os.path.join(self.output_dir, f"{output_prefix}_preprocessed.nii.gz"))
        
        # Save brain mask
        if 'brain_mask' in results:
            mask_img = nib.Nifti1Image(results['brain_mask'].astype(np.uint8), registered_affine)
            nib.save(mask_img, os.path.join(self.output_dir, f"{output_prefix}_brain_mask.nii.gz"))
        
        # Save intermediate results
        if 'denoised' in results:
            denoised_img = nib.Nifti1Image(results['denoised'], original_affine)
            nib.save(denoised_img, os.path.join(self.output_dir, f"{output_prefix}_denoised.nii.gz"))
        
        if 'bias_corrected' in results:
            bias_corrected_img = nib.Nifti1Image(results['bias_corrected'], original_affine)
            nib.save(bias_corrected_img, os.path.join(self.output_dir, f"{output_prefix}_bias_corrected.nii.gz"))
        
        if 'normalized' in results:
            normalized_img = nib.Nifti1Image(results['normalized'], original_affine)
            nib.save(normalized_img, os.path.join(self.output_dir, f"{output_prefix}_normalized.nii.gz"))
        
        if 'registered' in results:
            registered_img = nib.Nifti1Image(results['registered'], registered_affine)
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