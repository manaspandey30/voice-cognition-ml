# Voice-Based Cognitive Decline Pattern Detection

This project implements a proof-of-concept pipeline for detecting cognitive decline indicators through voice analysis. The system processes voice samples and extracts various features that might indicate early cognitive impairment.

## Features Analyzed

1. **Audio Features**
   - Mel-frequency cepstral coefficients (MFCCs)
   - Pitch variability
   - Speech tempo
   - Audio quality metrics

2. **Speech Patterns**
   - Pauses per sentence
   - Hesitation markers (uh, um, etc.)
   - Speech rate
   - Word count and complexity

3. **Cognitive Indicators**
   - Anomaly detection using clustering
   - Z-score analysis for outlier detection
   - Pattern recognition in speech flow

## Requirements

Install the required packages using:
```bash
pip install -r requirements.txt
```

## Usage

1. Place your audio files (WAV or MP3 format) in the project directory
2. Modify the `audio_files` list in `main()` to include your audio files
3. Run the script:
```bash
python cognitive_decline_detector.py
```

## Output

The script generates:
- Analysis results in the console
- Visualizations saved as 'analysis_results.png'
- Anomaly scores for each sample

## Methodology

1. **Audio Processing**
   - Convert audio to appropriate format
   - Extract acoustic features using librosa
   - Transcribe speech to text

2. **Feature Extraction**
   - Calculate speech patterns and metrics
   - Extract audio characteristics
   - Combine features into a comprehensive dataset

3. **Analysis**
   - Apply K-means clustering
   - Calculate anomaly scores
   - Generate visualizations

## Next Steps

1. **Clinical Validation**
   - Collaborate with neurologists for feature validation
   - Collect more diverse samples
   - Establish baseline metrics

2. **Technical Improvements**
   - Implement more sophisticated NLP features
   - Add real-time processing capabilities
   - Develop API endpoints for integration

3. **Research Directions**
   - Longitudinal studies
   - Cross-validation with other cognitive tests
   - Integration with other biomarkers

## Disclaimer

This is a proof-of-concept implementation and should not be used for clinical diagnosis without proper validation and medical supervision. 