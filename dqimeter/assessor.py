import cv2
import numpy as np
import onnxruntime as ort
import importlib.resources
import os

class DocumentQualityAssessor:
    """
    Assesses the quality of a document image by predicting various quality metrics.
    """
    def __init__(self, model_path=None):
        """
        Initializes the DocumentQualityAssessor.

        Args:
            model_path (str, optional): Path to the ONNX model file. If None,
                                        the default packaged model is used.
        """
        if model_path is None:
            with importlib.resources.path('dqimeter.resources', 'DQI_efficientnet_b0.onnx') as p:
                model_path = str(p)
        
        self.session = ort.InferenceSession(model_path)
        self.input_name = self.session.get_inputs()[0].name
        self._reg_keys = ['noise_level', 'blur_radius', 'brightness_factor', 'contrast_factor', 'jpeg_quality']

    def _preprocess(self, image_input):
        """Preprocesses the input image for the ONNX model."""
        if isinstance(image_input, str):
            if not os.path.exists(image_input):
                raise FileNotFoundError(f"Image file not found: {image_input}")
            img_bgr = cv2.imread(image_input)
            if img_bgr is None:
                raise ValueError(f"Failed to load image from {image_input}")
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        elif isinstance(image_input, np.ndarray):
            if len(image_input.shape) == 3 and image_input.shape[2] == 3:
                img_rgb = cv2.cvtColor(image_input, cv2.COLOR_BGR2RGB)
            else:
                img_rgb = image_input  # Assume already in RGB
        else:
            raise TypeError("image_input must be a file path (str) or a numpy array (CV2 image)")

        img_resized = cv2.resize(img_rgb, (224, 224))
        img_float = img_resized.astype(np.float32) / 255.0
        
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        
        img_normalized = (img_float - mean) / std
        img_chw = np.transpose(img_normalized, (2, 0, 1))
        return np.expand_dims(img_chw, axis=0)

    def assess(self, image, thresholds=None):
        """
        Runs inference on a document image to assess its quality.

        Args:
            image (str or numpy.ndarray): Path to the image or a loaded OpenCV image (BGR).
            thresholds (dict, optional): A dictionary to evaluate quality metrics against.
                                         Example:
                                         {
                                             'noise_level': {'max': 0.15},
                                             'blur_radius': {'max': 0.60},
                                             'jpeg_quality': {'min': 0.20},
                                             'lighting_artifact_detected': {'expected': False}
                                         }
        Returns:
            dict: A dictionary containing the raw quality scores and evaluation flags.
        """
        input_tensor = self._preprocess(image)
        outputs = self.session.run(None, {self.input_name: input_tensor})

        regression_out = outputs[0][0]
        classification_logits = outputs[1][0]
        predicted_class = int(np.argmax(classification_logits))

        reg_metrics = {key: float(val) for key, val in zip(self._reg_keys, regression_out)}

        results = {
            "regression_metrics": reg_metrics,
            "lighting_artifact_detected": bool(predicted_class),
            "raw_logits": classification_logits.tolist()
        }

        if thresholds:
            flags = {}
            for metric, config in thresholds.items():
                passed = True
                if metric in reg_metrics:
                    val = reg_metrics[metric]
                    if 'max' in config and val > config['max']:
                        passed = False
                    if 'min' in config and val < config['min']:
                        passed = False
                elif metric == 'lighting_artifact_detected':
                    val = results['lighting_artifact_detected']
                    if 'expected' in config and val != config['expected']:
                        passed = False
                flags[metric] = passed
            
            results['quality_flags'] = flags
            results['overall_pass'] = all(flags.values())

        return results
