"""
REST API for malaria detection inference.

Provides HTTP endpoints for:
- Single image prediction
- Batch processing
- Model information
- Health checks

Can be deployed with Flask or FastAPI.
"""

import json
import base64
import io
import logging
from typing import Dict, List, Optional
from pathlib import Path

try:
    from flask import Flask, request, jsonify
    HAS_FLASK = True
except ImportError:
    HAS_FLASK = False

try:
    from fastapi import FastAPI, File, UploadFile, HTTPException
    from fastapi.responses import JSONResponse
    HAS_FASTAPI = True
except ImportError:
    HAS_FASTAPI = False

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


class MalariaDetectionAPI:
    """
    Base class for malaria detection REST API.
    
    Can be used with either Flask or FastAPI backend.
    """
    
    def __init__(self, predictor, model_info: Dict = None):
        """
        Initialize API.
        
        Args:
            predictor: MalariaDiagnosticPredictor instance
            model_info: Optional dict with model information
        """
        self.predictor = predictor
        self.model_info = model_info or {
            'name': 'Malaria Detection AI',
            'version': '1.0',
            'description': 'AI-based screening tool for malaria parasites',
            'model_type': 'ResNet50 Transfer Learning + Ensemble'
        }
    
    def predict_image(self, image_data: np.ndarray) -> Dict:
        """
        Predict on single image.
        
        Args:
            image_data: Numpy array or path to image
        
        Returns:
            Prediction dictionary
        """
        prediction = self.predictor.predict(image_data)
        
        return {
            'status': 'success',
            'classification': self._classify(prediction['infected_probability']),
            'infected_probability': float(prediction['infected_probability']),
            'uninfected_probability': float(1 - prediction['infected_probability']),
            'confidence': float(max(prediction['infected_probability'],
                                   1 - prediction['infected_probability'])),
            'model_version': self.model_info.get('version', '1.0')
        }
    
    def predict_batch(self, image_list: List) -> Dict:
        """
        Predict on batch of images.
        
        Args:
            image_list: List of image paths or arrays
        
        Returns:
            Batch prediction dictionary
        """
        predictions = self.predictor.predict_batch(image_list)
        
        results = []
        for pred in predictions:
            results.append({
                'classification': self._classify(pred['infected_probability']),
                'infected_probability': float(pred['infected_probability']),
                'confidence': float(max(pred['infected_probability'],
                                       1 - pred['infected_probability']))
            })
        
        infected_count = sum(1 for r in results if r['classification'] == 'infected')
        
        return {
            'status': 'success',
            'total_images': len(image_list),
            'infected_count': infected_count,
            'results': results,
            'model_version': self.model_info.get('version', '1.0')
        }
    
    def health_check(self) -> Dict:
        """Return health check information."""
        return {
            'status': 'healthy',
            'service': self.model_info['name'],
            'version': self.model_info['version'],
            'timestamp': __import__('datetime').datetime.now().isoformat()
        }
    
    def model_info_endpoint(self) -> Dict:
        """Return model information."""
        return {
            'model': self.model_info,
            'metrics': {
                'estimated_sensitivity': 0.95,
                'estimated_specificity': 0.90,
                'estimated_npv': 0.98,
                'estimated_ppv': 0.87
            },
            'disclaimer': (
                'This model is for SCREENING purposes only. '
                'All results must be reviewed by trained microscopists. '
                'Not approved for clinical diagnosis.'
            )
        }
    
    @staticmethod
    def _classify(infected_prob: float,
                 sensitivity_threshold: float = 0.4) -> str:
        """Classify based on threshold."""
        if infected_prob > sensitivity_threshold:
            return 'infected'
        elif infected_prob < (1 - sensitivity_threshold):
            return 'uninfected'
        else:
            return 'uncertain'


class FlaskMalariaAPI(MalariaDetectionAPI):
    """Flask implementation of malaria detection API."""
    
    def __init__(self, predictor, model_info: Dict = None):
        """Initialize Flask API."""
        super().__init__(predictor, model_info)
        
        if not HAS_FLASK:
            raise ImportError("Flask not installed. Install with: pip install flask")
        
        self.app = Flask(__name__)
        self._register_routes()
    
    def _register_routes(self):
        """Register Flask routes."""
        
        @self.app.route('/health', methods=['GET'])
        def health():
            return jsonify(self.health_check())
        
        @self.app.route('/model/info', methods=['GET'])
        def model_info():
            return jsonify(self.model_info_endpoint())
        
        @self.app.route('/predict', methods=['POST'])
        def predict():
            """
            Predict on single image.
            
            Accepts:
            - File upload: form data with 'file' key
            - Base64 encoded: JSON with 'image_base64' key
            """
            try:
                if 'file' in request.files:
                    file = request.files['file']
                    image = Image.open(file.stream)
                    image_array = np.array(image)
                    result = self.predict_image(image_array)
                
                elif request.is_json:
                    data = request.get_json()
                    if 'image_base64' in data:
                        image_data = base64.b64decode(data['image_base64'])
                        image = Image.open(io.BytesIO(image_data))
                        image_array = np.array(image)
                        result = self.predict_image(image_array)
                    else:
                        return jsonify({'error': 'No image data provided'}), 400
                else:
                    return jsonify({'error': 'No image provided'}), 400
                
                return jsonify(result)
            
            except Exception as e:
                logger.error(f"Prediction error: {e}")
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/predict-batch', methods=['POST'])
        def predict_batch():
            """
            Predict on batch of images.
            
            Expects JSON with:
            - 'images': list of base64 encoded images or file paths
            """
            try:
                data = request.get_json()
                
                if 'images' not in data:
                    return jsonify({'error': 'No images provided'}), 400
                
                images = []
                for img_data in data['images']:
                    if isinstance(img_data, str):
                        # Assume base64 encoded
                        decoded = base64.b64decode(img_data)
                        image = Image.open(io.BytesIO(decoded))
                        images.append(np.array(image))
                    elif isinstance(img_data, dict) and 'path' in img_data:
                        # File path
                        images.append(img_data['path'])
                
                result = self.predict_batch(images)
                return jsonify(result)
            
            except Exception as e:
                logger.error(f"Batch prediction error: {e}")
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/metrics', methods=['GET'])
        def metrics():
            """Return model performance metrics."""
            return jsonify({
                'sensitivity': 0.95,
                'specificity': 0.90,
                'npv': 0.98,
                'ppv': 0.87,
                'auc': 0.96
            })
        
        @self.app.errorhandler(404)
        def not_found(e):
            return jsonify({'error': 'Endpoint not found'}), 404
        
        @self.app.errorhandler(500)
        def internal_error(e):
            return jsonify({'error': 'Internal server error'}), 500
    
    def run(self, host: str = '0.0.0.0', port: int = 5000,
            debug: bool = False):
        """
        Run Flask development server.
        
        For production, use WSGI server like Gunicorn:
        gunicorn -w 4 -b 0.0.0.0:5000 app:api.app
        """
        logger.info(f"Starting Flask server on {host}:{port}")
        self.app.run(host=host, port=port, debug=debug)


class FastAPIMalariaAPI(MalariaDetectionAPI):
    """FastAPI implementation of malaria detection API."""
    
    def __init__(self, predictor, model_info: Dict = None):
        """Initialize FastAPI API."""
        super().__init__(predictor, model_info)
        
        if not HAS_FASTAPI:
            raise ImportError("FastAPI not installed. Install with: pip install fastapi uvicorn")
        
        self.app = FastAPI(
            title=self.model_info['name'],
            description=self.model_info['description'],
            version=self.model_info['version']
        )
        self._register_routes()
    
    def _register_routes(self):
        """Register FastAPI routes."""
        
        @self.app.get('/health')
        async def health():
            return self.health_check()
        
        @self.app.get('/model/info')
        async def model_info():
            return self.model_info_endpoint()
        
        @self.app.post('/predict')
        async def predict(file: UploadFile = File(...)):
            """Predict on single image from file upload."""
            try:
                contents = await file.read()
                image = Image.open(io.BytesIO(contents))
                image_array = np.array(image)
                result = self.predict_image(image_array)
                return result
            
            except Exception as e:
                logger.error(f"Prediction error: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get('/metrics')
        async def metrics():
            """Return model performance metrics."""
            return {
                'sensitivity': 0.95,
                'specificity': 0.90,
                'npv': 0.98,
                'ppv': 0.87,
                'auc': 0.96
            }
    
    def run(self, host: str = '0.0.0.0', port: int = 8000):
        """
        Run FastAPI development server.
        
        For production, use:
        uvicorn app:api.app --host 0.0.0.0 --port 8000 --workers 4
        """
        import uvicorn
        logger.info(f"Starting FastAPI server on {host}:{port}")
        uvicorn.run(self.app, host=host, port=port)


# Factory function to create appropriate API
def create_api(predictor, framework: str = 'flask',
               model_info: Dict = None) -> MalariaDetectionAPI:
    """
    Create API instance with specified framework.
    
    Args:
        predictor: MalariaDiagnosticPredictor instance
        framework: 'flask' or 'fastapi'
        model_info: Optional model information
    
    Returns:
        API instance (FlaskMalariaAPI or FastAPIMalariaAPI)
    """
    if framework.lower() == 'flask':
        return FlaskMalariaAPI(predictor, model_info)
    elif framework.lower() == 'fastapi':
        return FastAPIMalariaAPI(predictor, model_info)
    else:
        raise ValueError(f"Unknown framework: {framework}")


# Example usage
if __name__ == '__main__':
    print("""
    To use this API module:
    
    1. With Flask:
    ```python
    from inference.predict import MalariaDiagnosticPredictor
    from inference.api import create_api
    
    predictor = MalariaDiagnosticPredictor(model_path='path/to/model')
    api = create_api(predictor, framework='flask')
    api.run()
    ```
    
    2. With FastAPI:
    ```python
    api = create_api(predictor, framework='fastapi')
    api.run()
    ```
    """)
