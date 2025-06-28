#!/usr/bin/env python3
"""
Example usage of the MRI Preprocessing Pipeline
==============================================

This script demonstrates how to use the preprocessing pipeline
with different configurations and options.
"""

from mri_preprocessing_pipeline import MRIPreprocessingPipeline
import os

def example_single_image_processing():
    """
    Example: Process a single image with custom parameters.
    """
    print("=== Single Image Processing Example ===")
    
    # Initialize pipeline
    pipeline = MRIPreprocessingPipeline(
        input_dir="IXI-T1",
        template_dir="templates",
        output_dir="preprocessed"
    )
    
    # Get the first available image
    input_files = [f for f in os.listdir("IXI-T1") if f.endswith('.nii.gz')]
    if not input_files:
        print("No input files found!")
        return
    
    input_file = os.path.join("IXI-T1", input_files[0])
    output_prefix = input_files[0].replace('.nii.gz', '')
    
    print(f"Processing: {input_file}")
    
    # Process the image
    results = pipeline.preprocess_single_image(input_file, output_prefix)
    
    print(f"Processing completed. Results saved with prefix: {output_prefix}")
    print(f"Output directory: {pipeline.output_dir}")


def example_custom_processing():
    """
    Example: Custom processing with different parameters.
    """
    print("\n=== Custom Processing Example ===")
    
    # Initialize pipeline
    pipeline = MRIPreprocessingPipeline()
    
    # Load an image
    input_files = [f for f in os.listdir("IXI-T1") if f.endswith('.nii.gz')]
    if not input_files:
        print("No input files found!")
        return
    
    input_file = os.path.join("IXI-T1", input_files[0])
    img = pipeline.load_image(input_file)
    img_data = img.get_fdata()
    
    print("Applying custom processing steps...")
    
    # Custom denoising with total variation
    denoised = pipeline.denoise_image(img_data, method='tv')
    print("✓ Denoising completed (TV method)")
    
    # Custom bias field estimation with higher degree polynomial
    bias_field = pipeline.estimate_bias_field(denoised, degree=4)
    bias_corrected = pipeline.correct_bias_field(denoised, bias_field)
    print("✓ Bias field correction completed (degree=4)")
    
    # Custom intensity normalization with histogram equalization
    normalized = pipeline.normalize_intensity(bias_corrected, method='histogram')
    print("✓ Intensity normalization completed (histogram)")
    
    # Custom skull stripping with watershed method
    stripped, brain_mask = pipeline.skull_stripping(normalized, method='watershed')
    print("✓ Skull stripping completed (watershed)")
    
    # Save custom results
    output_prefix = "custom_processing"
    custom_results = {
        'original': img_data,
        'denoised': denoised,
        'bias_corrected': bias_corrected,
        'normalized': normalized,
        'stripped': stripped,
        'brain_mask': brain_mask,
        'affine': img.affine
    }
    
    pipeline.save_results(custom_results, output_prefix)
    pipeline.create_visualizations(custom_results, output_prefix)
    
    print(f"Custom processing completed. Results saved with prefix: {output_prefix}")


def example_batch_processing():
    """
    Example: Process multiple images in batch.
    """
    print("\n=== Batch Processing Example ===")
    
    # Initialize pipeline
    pipeline = MRIPreprocessingPipeline()
    
    # Process all images
    print("Starting batch processing...")
    pipeline.process_all_images()
    
    print("Batch processing completed!")


def example_quality_control():
    """
    Example: Quality control and visualization.
    """
    print("\n=== Quality Control Example ===")
    
    # Initialize pipeline
    pipeline = MRIPreprocessingPipeline()
    
    # Load an image
    input_files = [f for f in os.listdir("IXI-T1") if f.endswith('.nii.gz')]
    if not input_files:
        print("No input files found!")
        return
    
    input_file = os.path.join("IXI-T1", input_files[0])
    img = pipeline.load_image(input_file)
    img_data = img.get_fdata()
    
    print("Creating quality control visualizations...")
    
    # Create bias field visualization
    bias_field = pipeline.estimate_bias_field(img_data)
    bias_corrected = pipeline.correct_bias_field(img_data, bias_field)
    
    pipeline.visualize_bias_field(
        img_data, 
        bias_field, 
        bias_corrected, 
        "quality_control_bias_field.png"
    )
    
    print("✓ Bias field visualization saved as 'quality_control_bias_field.png'")
    
    # Create pipeline overview
    denoised = pipeline.denoise_image(img_data)
    normalized = pipeline.normalize_intensity(bias_corrected)
    registered, registered_affine = pipeline.register_to_template(normalized, img.affine)
    stripped, brain_mask = pipeline.skull_stripping(registered)
    
    qc_results = {
        'original': img_data,
        'denoised': denoised,
        'bias_corrected': bias_corrected,
        'normalized': normalized,
        'registered': registered,
        'stripped': stripped
    }
    
    pipeline.create_pipeline_overview(qc_results, "quality_control")
    
    print("✓ Pipeline overview saved as 'quality_control_pipeline_overview.png'")


def main():
    """
    Run all examples.
    """
    print("MRI Preprocessing Pipeline - Example Usage")
    print("=" * 50)
    
    # Check if input directory exists
    if not os.path.exists("IXI-T1"):
        print("Error: IXI-T1 directory not found!")
        print("Please ensure you have the input data in the IXI-T1 directory.")
        return
    
    # Check if template directory exists
    if not os.path.exists("templates"):
        print("Error: templates directory not found!")
        print("Please ensure you have the MNI152 template files in the templates directory.")
        return
    
    try:
        # Run examples
        example_single_image_processing()
        example_custom_processing()
        example_quality_control()
        
        # Uncomment the following line to run batch processing
        # example_batch_processing()
        
        print("\n" + "=" * 50)
        print("All examples completed successfully!")
        print("Check the 'preprocessed' directory for results.")
        
    except Exception as e:
        print(f"Error running examples: {str(e)}")
        print("Please check that all dependencies are installed:")
        print("pip install -r requirements.txt")


if __name__ == "__main__":
    main() 