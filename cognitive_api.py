import os
from cognitive_decline_detector import CognitiveDeclineDetector
import pandas as pd
import tempfile
from flask import Flask, request, jsonify
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize the detector
detector = CognitiveDeclineDetector()

# Risk scoring thresholds
RISK_THRESHOLDS = {
    'low': 0.8,
    'moderate': 1.5,
    'high': 2.0
}

def score_audio_file(audio_path):
    """
    Score an audio file for cognitive decline risk
    
    Args:
        audio_path (str): Path to the audio file
        
    Returns:
        dict: Risk score results including:
            - raw_score: The raw anomaly score
            - risk_level: Categorical risk level (low, moderate, high)
            - features: Key features that contributed to the score
    """
    try:
        # Process the audio file
        features = detector.process_audio_file(audio_path)
        if not features:
            return {
                'error': 'Failed to process audio file',
                'success': False
            }
        
        # Convert to DataFrame for analysis
        df = pd.DataFrame([features])
        
        # Apply anomaly detection
        results = detector.detect_anomalies(df)
        
        # Extract the anomaly score
        anomaly_score = results['anomaly_score'].iloc[0]
        
        # Determine risk level
        if anomaly_score < RISK_THRESHOLDS['low']:
            risk_level = 'low'
        elif anomaly_score < RISK_THRESHOLDS['moderate']:
            risk_level = 'moderate'
        elif anomaly_score < RISK_THRESHOLDS['high']:
            risk_level = 'high'
        else:
            risk_level = 'very high'
        
        # Get top contributing features
        contributing_features = {}
        numeric_cols = df.select_dtypes(include=['number']).columns
        
        # Get silence and energy features if available
        key_indicators = ['silence_ratio', 'silence_count', 'energy_var', 'energy_changes']
        for indicator in key_indicators:
            if indicator in df.columns:
                contributing_features[indicator] = float(df[indicator].iloc[0])
        
        return {
            'success': True,
            'raw_score': float(anomaly_score),
            'risk_level': risk_level,
            'features': contributing_features
        }
        
    except Exception as e:
        logger.error(f"Error scoring audio file: {e}")
        return {
            'error': str(e),
            'success': False
        }

# Flask API
app = Flask(__name__)

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({'status': 'healthy'})

@app.route('/score', methods=['POST'])
def score_audio():
    """Score an audio file for cognitive decline risk"""
    # Check if file was uploaded
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded', 'success': False}), 400
    
    file = request.files['file']
    
    # Check if filename is empty
    if file.filename == '':
        return jsonify({'error': 'Empty filename', 'success': False}), 400
    
    # Save temporarily and process
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as temp:
        file.save(temp.name)
        temp_path = temp.name
    
    try:
        # Process the file
        result = score_audio_file(temp_path)
        # Clean up the temp file
        os.unlink(temp_path)
        return jsonify(result)
    except Exception as e:
        # Clean up the temp file
        if os.path.exists(temp_path):
            os.unlink(temp_path)
        return jsonify({'error': str(e), 'success': False}), 500

def run_api(host='0.0.0.0', port=5000, debug=False):
    """Run the Flask API"""
    app.run(host=host, port=port, debug=debug)

if __name__ == '__main__':
    # Run the API server
    run_api(debug=True) 