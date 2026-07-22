import os
import random
import json
import numpy as np
from PIL import Image
from dqimeter import DocumentQualityAssessor

# This test script requires the 'datasets' library.
# You can install it with: pip install datasets
try:
    from datasets import load_dataset
except ImportError:
    print("Error: The 'datasets' library is required to run this test script.")
    print("Please install it using: pip install datasets")
    exit()

def calculate_divergence_report(results):
    """Calculates and prints the final performance report."""
    
    # --- Classification Accuracy ---
    class_correct = sum(1 for r in results if r['pred_lighting'] == r['gt_lighting'])
    class_accuracy = class_correct / len(results)
    
    # --- Regression Mean Absolute Error (MAE) ---
    mae_errors = {
        'noise_level': [], 'blur_radius': [], 'brightness_factor': [],
        'contrast_factor': [], 'jpeg_quality': []
    }
    
    for r in results:
        for key in mae_errors.keys():
            # The model's jpeg_quality is normalized 0-1, ground truth is 0-100
            gt_val = r[f'gt_{key}'] / 100.0 if key == 'jpeg_quality' else r[f'gt_{key}']
            pred_val = r['pred_metrics'][key]
            mae_errors[key].append(abs(gt_val - pred_val))
            
    print("\n--- DQIMeter Performance Report ---")
    print(f"Tested on {len(results)} random samples from the validation set.")
    
    print(f"\n1. Lighting Artifact Detection (Classification)")
    print(f"   - Accuracy: {class_accuracy:.2%}")
    
    print("\n2. Quality Metrics (Regression Divergence)")
    print("   - Mean Absolute Error (MAE):")
    for key, errors in mae_errors.items():
        avg_mae = np.mean(errors)
        # Scale jpeg quality back to % for interpretability
        if key == 'jpeg_quality':
            print(f"     - {key}: {avg_mae * 100:.2f}%")
        else:
            print(f"     - {key}: {avg_mae:.4f}")
    print("\n------------------------------------")


def run_evaluation(num_samples=100):
    """
    Loads the dataset, runs evaluation on a subset of samples, and generates a report.
    """
    print("Loading validation dataset 'racineai/ocr-pdf-degraded'...")
    try:
        # Using streaming=True to avoid downloading the full dataset at once
        dataset = load_dataset("racineai/ocr-pdf-degraded", split="train", streaming=True)
    except Exception as e:
        print(f"Failed to load dataset. Please check your internet connection. Error: {e}")
        return
        
    print(f"Dataset loaded. Preparing to test on {num_samples} samples...")
    
    assessor = DocumentQualityAssessor()
    evaluation_results = []
    
    # Take a number of samples from the streaming dataset
    samples = list(dataset.take(num_samples))

    for i, item in enumerate(samples):
        if (i + 1) % 10 == 0:
            print(f"  Processing sample {i + 1}/{num_samples}...")
        
        # The 'image' field is a PIL Image
        pil_image = item['image']
        # Convert PIL Image to OpenCV format (BGR numpy array)
        cv_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
        
        # Get ground truth from the 'params' string
        params = json.loads(item['params'])
        
        # Run assessment
        report = assessor.assess(cv_image)
        
        evaluation_results.append({
            'pred_metrics': report['regression_metrics'],
            'pred_lighting': report['lighting_artifact_detected'],
            'gt_noise_level': params.get('noise_level', 0.0),
            'gt_blur_radius': params.get('blur_radius', 0.0),
            'gt_brightness_factor': params.get('brightness_factor', 1.0),
            'gt_contrast_factor': params.get('contrast_factor', 1.0),
            'gt_jpeg_quality': params.get('jpeg_quality', 100.0),
            'gt_lighting': params.get('apply_lighting', False)
        })

    # Generate and print the final report
    calculate_divergence_report(evaluation_results)

if __name__ == "__main__":
    run_evaluation(num_samples=100)
