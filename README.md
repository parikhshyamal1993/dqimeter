# 🛡️ DQIMeter: Your First Line of Defense in Document AI

**DQIMeter** is a production-grade Python SDK that provides an essential **Document Quality Index (DQI)**. In any automated document processing pipeline (KYC, OCR, invoice processing), the greatest point of failure is the quality of the input image. **Garbage In, Garbage Out.**

DQIMeter intercepts low-quality documents at the point of entry, providing instant, actionable feedback *before* they consume expensive downstream resources.

---
## 🚀 Key Features

*   **Multi-Task Deep Learning**: A single, efficient model simultaneously predicts multiple document quality dimensions.
*   **Comprehensive Quality Metrics**:
    *   **Regression**: Noise Level, Blur Radius, Brightness, Contrast, JPEG Compression Artifacts.
    *   **Classification**: Detects the presence of severe lighting glare or shadows.
*   **Actionable Insights**: Use the raw scores to understand *why* a document is low quality and provide specific feedback to the user (e.g., "Image is too blurry," "Please move to a brighter room").
*   **Configurable Quality Gates**: Define your own pass/fail thresholds to create a robust quality gate tailored to your business rules.
*   **High-Performance & Lightweight**: Built on an **EfficientNet-B0** backbone and deployed with the ONNX runtime for blazing-fast, low-latency inference on CPU or GPU.
*   **Developer-First API**: A clean, simple interface that accepts file paths or in-memory OpenCV image arrays.

---
## 🏗️ The Engineering Behind DQIMeter

### The "Garbage In, Garbage Out" Problem
Downstream models for OCR, signature detection, or data extraction are sensitive to image quality. A blurry, dark, or compressed image of a driver's license will cause these models to fail, leading to wasted GPU cycles and a frustrating user experience. DQIMeter solves this by acting as an intelligent gatekeeper.

### Architecture: EfficientNet-B0 Multi-Task Model
We chose **EfficientNet-B0** for its optimal balance of high accuracy and parameter efficiency, making it perfect for a low-latency gatekeeper service.

The model was engineered with a shared backbone that splits into two specialized heads:
1.  **Regression Head**: Predicts the continuous values of five key degradation metrics.
2.  **Classification Head**: Performs a binary classification to detect the presence of lighting artifacts.

This multi-task architecture provides a holistic quality assessment from a single, fast inference pass.

### The Data Engine: `racineai/ocr-pdf-degraded`
The model was trained on the `racineai/ocr-pdf-degraded` dataset, which programmatically applies realistic document degradations. This provides precise, objective ground-truth labels for quality metrics, allowing the model to learn the underlying physics of document degradation far more accurately than subjective human labeling would allow.

---
## 🛠️ Installation

DQIMeter will be available on PyPI.

```bash
# From PyPI (Coming Soon)
pip install dqimeter

# Or directly from the repository
pip install git+https://github.com/parikhshyamal1993/dqimeter.git
```

---
## 💻 Quick Start

Using DQIMeter is simple and intuitive.

```python
from dqimeter import DocumentQualityAssessor
import cv2

# Initialize the assessor (loads the model automatically)
assessor = DocumentQualityAssessor()

# 1. Get a full quality report for an image
image_path = "path/to/document.jpg"
report = assessor.assess(image_path)

print("--- Full Quality Report ---")
print(f"Lighting Artifact Detected: {report['lighting_artifact_detected']}")
print("Regression Metrics:")
for metric, value in report['regression_metrics'].items():
    print(f"  - {metric}: {value:.4f}")

# 2. Use it as a Quality Gate with custom rules
quality_rules = {
    'noise_level': {'max': 0.15},          # Must be <= 0.15
    'blur_radius': {'max': 0.55},          # Must be <= 0.55
    'jpeg_quality': {'min': 0.30},         # Must be >= 0.30 (30%)
    'lighting_artifact_detected': {'expected': False} # Must not have lighting issues
}

gate_report = assessor.assess(image_path, thresholds=quality_rules)

print("\n--- Quality Gate Report ---")
if gate_report['overall_pass']:
    print("✅ Document PASSED the quality check.")
else:
    print("❌ Document FAILED the quality check.")
    # Provide specific feedback to the user
    for metric, passed in gate_report['quality_flags'].items():
        if not passed:
            print(f"  - Reason: Failed '{metric}' check.")

```

---
## 🔮 Future Roadmap

DQIMeter is the foundational component for building robust document intelligence systems. Future enhancements include:
*   **Presentation Attack Detection (PAD)**: Expanding the classification head to detect screen spoofing attacks (e.g., photos of documents on a monitor).
*   **Moiré Pattern Detection**: Identifying digital interference patterns common in screen captures.
*   **Auto-Enhancement Suggestions**: Recommending specific `docscan` enhancements based on the quality report.

---
## 📄 License
This project is licensed under the MIT License.
