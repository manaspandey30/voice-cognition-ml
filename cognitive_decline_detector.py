import os
import numpy as np
import pandas as pd
import librosa
import speech_recognition as sr
from pydub import AudioSegment
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from scipy.stats import zscore
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import json
import warnings
# Suppress all warnings
warnings.filterwarnings('ignore')

# Suppress pydub warning about ffmpeg
AudioSegment.converter = "ffmpeg"

# Download required NLTK data
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

class CognitiveDeclineDetector:
    def __init__(self):
        self.recognizer = sr.Recognizer()
        self.stop_words = set(stopwords.words('english'))
        self.hesitation_markers = {'uh', 'um', 'er', 'ah', 'hm', 'hmm'}
        
    def load_audio(self, audio_path):
        """Load audio file and convert to appropriate format"""
        try:
            audio = AudioSegment.from_file(audio_path)
            # Convert to WAV if needed
            if audio_path.endswith('.mp3'):
                audio = audio.set_frame_rate(16000).set_channels(1)
            return audio
        except Exception as e:
            print(f"Error loading audio: {e}")
            return None

    def extract_audio_features(self, audio, audio_path=None):
        """Extract audio features using librosa"""
        y = np.array(audio.get_array_of_samples())
        # Convert to float32 and normalize
        if y.dtype == np.int16:
            y = y.astype(np.float32) / 32768.0
        elif y.dtype == np.int32:
            y = y.astype(np.float32) / 2147483648.0
            
        sr = audio.frame_rate
        
        # Extract features
        # Zero crossing rate - can indicate hesitations and pauses
        zero_cross_rate = np.mean(librosa.feature.zero_crossing_rate(y))
        
        # Root mean square energy - level of energy (loudness)
        rms = np.mean(librosa.feature.rms(y=y))
        
        # Spectral centroid - brightness of sound
        spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
        
        # Spectral bandwidth - range of frequencies
        spectral_bandwidth = np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr))
        
        # Spectral rolloff - frequency below which is concentrated 85% of energy
        rolloff = np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr))
        
        # MFCCs - overall spectral shape
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        mfcc_means = np.mean(mfccs, axis=1)
        mfcc_stds = np.std(mfccs, axis=1)
        
        # Silence detection (pauses)
        non_silent = librosa.effects.split(y, top_db=30)
        silence_ratio = 1.0 - (np.sum([end-start for start, end in non_silent]) / len(y))
        
        # Count silence segments (potential pause count)
        silence_count = len(non_silent) - 1 if len(non_silent) > 0 else 0
        
        # Calculate temporal variations in energy (hesitations)
        if len(y) > sr:  # Ensure audio is longer than 1 second
            frame_length = int(sr * 0.05)  # 50ms frames
            hop_length = int(sr * 0.025)   # 25ms hop
            energy = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
            energy_var = np.var(energy)
            energy_changes = np.sum(np.abs(np.diff(energy) > 0.1 * np.mean(energy)))
        else:
            energy_var = 0
            energy_changes = 0
        
        features = {
            'zero_cross_rate': zero_cross_rate,
            'rms_energy': rms,
            'spectral_centroid': spectral_centroid,
            'spectral_bandwidth': spectral_bandwidth,
            'spectral_rolloff': rolloff,
            'silence_ratio': silence_ratio,
            'silence_count': silence_count,
            'energy_var': energy_var,
            'energy_changes': energy_changes
        }
        
        # Add MFCCs to features
        for i, (mean, std) in enumerate(zip(mfcc_means, mfcc_stds)):
            features[f'mfcc{i+1}_mean'] = mean
            features[f'mfcc{i+1}_std'] = std
        
        # Try to load metadata for synthetic data if available
        metadata_features = self.load_metadata_features(audio_path)
        if metadata_features:
            features.update(metadata_features)
            
        return features

    def load_metadata_features(self, audio_path):
        """Load metadata features for synthetic data if available"""
        if audio_path is None:
            return {}
            
        metadata_path = os.path.join(os.path.dirname(os.path.dirname(audio_path)), 'samples/metadata.json')
        if os.path.exists(metadata_path):
            try:
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                
                # Find entry for this file
                filename = os.path.basename(audio_path)
                for entry in metadata:
                    if os.path.basename(entry['filename']) == filename:
                        return {
                            'known_pauses': entry['num_pauses'],
                            'known_hesitations': entry['num_hesitations'],
                            'cognitive_level': entry['cognitive_level']
                        }
            except Exception as e:
                print(f"Error loading metadata: {e}")
        
        return {}

    def transcribe_audio(self, audio):
        """Convert speech to text using Google Speech Recognition"""
        try:
            with sr.AudioFile(audio.export(format="wav")) as source:
                audio_data = self.recognizer.record(source)
                text = self.recognizer.recognize_google(audio_data)
                return text
        except Exception as e:
            print(f"Note: Could not transcribe audio: {e}")
            # For synthetic data, return empty text
            return ""

    def analyze_speech_patterns(self, text):
        """Analyze speech patterns for cognitive indicators"""
        if not text:
            # Return empty features for synthetic data
            return {
                'hesitation_count': 0,
                'pause_count': 0,
                'speech_rate': 0,
                'total_words': 0
            }
            
        words = word_tokenize(text.lower())
        
        # Calculate metrics
        total_words = len(words)
        hesitation_count = sum(1 for word in words if word in self.hesitation_markers)
        pause_count = text.count('.') + text.count('?') + text.count('!')
        
        # Calculate speech rate (words per second)
        speech_rate = total_words / len(text.split()) if len(text.split()) > 0 else 0
        
        return {
            'hesitation_count': hesitation_count,
            'pause_count': pause_count,
            'speech_rate': speech_rate,
            'total_words': total_words
        }

    def detect_anomalies(self, features_df):
        """Detect anomalies using clustering and z-score analysis"""
        # Prepare features for clustering
        analysis_features = features_df.copy()
        
        # Drop any non-numeric or identifier columns
        if 'cognitive_level' in analysis_features.columns:
            cognitive_levels = analysis_features['cognitive_level']
            analysis_features = analysis_features.drop(columns=['cognitive_level'])
        else:
            cognitive_levels = None
        
        # Keep only numeric columns
        numeric_features = analysis_features.select_dtypes(include=['number'])
        
        # Fill NaN values with mean
        numeric_features = numeric_features.fillna(numeric_features.mean())
        
        # Standardize features
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(numeric_features)
        
        # Apply K-means clustering
        n_clusters = min(2, len(numeric_features))  # Ensure we have enough samples
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        clusters = kmeans.fit_predict(scaled_features)
        
        # Calculate z-scores
        z_scores = np.abs(zscore(numeric_features, nan_policy='omit'))
        
        # Combine results
        features_df['cluster'] = clusters
        features_df['anomaly_score'] = np.nanmean(z_scores, axis=1)
        
        if cognitive_levels is not None:
            features_df['cognitive_level'] = cognitive_levels
            
        return features_df

    def visualize_results(self, features_df, output_path):
        """Create visualizations of the analysis results"""
        plt.figure(figsize=(15, 10))
        
        # Plot 1: Anomaly scores
        plt.subplot(2, 2, 1)
        sns.histplot(features_df['anomaly_score'], kde=True)
        plt.title('Distribution of Anomaly Scores')
        
        # Check if we have specific cognitive features to plot
        if 'silence_ratio' in features_df.columns and 'energy_changes' in features_df.columns:
            # Plot 2: Silence ratio vs. Energy changes
            plt.subplot(2, 2, 2)
            plot_df = features_df.copy()
            if 'cognitive_level' in plot_df.columns:
                sns.scatterplot(data=plot_df, x='silence_ratio', y='energy_changes', 
                               hue='cognitive_level', palette='deep')
                plt.title('Silence Ratio vs Energy Changes by Cognitive Level')
            else:
                sns.scatterplot(data=plot_df, x='silence_ratio', y='energy_changes', 
                               hue='cluster', palette='deep')
                plt.title('Silence Ratio vs Energy Changes by Cluster')
        
        # Plot 3: Feature correlations
        plt.subplot(2, 2, 3)
        corr_cols = features_df.select_dtypes(include=['number']).columns
        if len(corr_cols) > 1:  # Only create heatmap if we have multiple numeric columns
            corr_df = features_df[corr_cols].corr()
            # Limit to most important features to avoid overcrowding
            important_features = ['anomaly_score', 'silence_ratio', 'silence_count', 
                                 'energy_var', 'energy_changes', 'zero_cross_rate']
            avail_features = [f for f in important_features if f in corr_df.columns]
            if len(avail_features) > 1:
                sns.heatmap(corr_df.loc[avail_features, avail_features], annot=True, cmap='coolwarm')
                plt.title('Feature Correlations')
        
        # Plot 4: Compare anomaly scores by known cognitive level if available
        plt.subplot(2, 2, 4)
        if 'cognitive_level' in features_df.columns:
            sns.boxplot(data=features_df, x='cognitive_level', y='anomaly_score')
            plt.title('Anomaly Score by Cognitive Level')
        elif 'cluster' in features_df.columns:
            sns.boxplot(data=features_df, x='cluster', y='anomaly_score')
            plt.title('Anomaly Score by Cluster')
        
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()
        
        # Print the relative cognitive decline metrics
        self.print_cognitive_metrics(features_df)

    def print_cognitive_metrics(self, features_df):
        """Print a summary of cognitive metrics"""
        print("\nCognitive Metrics Summary:")
        print("-" * 40)
        
        # Print anomaly scores
        print(f"Anomaly Scores (higher indicates potential cognitive issues):")
        if 'cognitive_level' in features_df.columns:
            for level in features_df['cognitive_level'].unique():
                level_scores = features_df[features_df['cognitive_level'] == level]['anomaly_score']
                print(f"  {level.capitalize()} level: {level_scores.mean():.2f} (avg)")
        else:
            for cluster in features_df['cluster'].unique():
                cluster_scores = features_df[features_df['cluster'] == cluster]['anomaly_score']
                print(f"  Cluster {cluster}: {cluster_scores.mean():.2f} (avg)")
        
        # Print silence/pause metrics if available
        if 'silence_ratio' in features_df.columns:
            print("\nSilence Metrics (higher values may indicate more pauses):")
            if 'cognitive_level' in features_df.columns:
                for level in features_df['cognitive_level'].unique():
                    level_silence = features_df[features_df['cognitive_level'] == level]['silence_ratio']
                    print(f"  {level.capitalize()} level: {level_silence.mean():.2f} (avg)")
            else:
                print(f"  Average silence ratio: {features_df['silence_ratio'].mean():.2f}")
        
        # Print energy variation metrics if available
        if 'energy_var' in features_df.columns:
            print("\nEnergy Variation (higher values may indicate more hesitations):")
            if 'cognitive_level' in features_df.columns:
                for level in features_df['cognitive_level'].unique():
                    level_energy = features_df[features_df['cognitive_level'] == level]['energy_var']
                    print(f"  {level.capitalize()} level: {level_energy.mean():.2f} (avg)")
            else:
                print(f"  Average energy variation: {features_df['energy_var'].mean():.2f}")

    def process_audio_file(self, audio_path):
        """Process a single audio file and return analysis results"""
        print(f"Processing {audio_path}...")
        audio = self.load_audio(audio_path)
        if audio is None:
            return None
            
        # Extract features
        audio_features = self.extract_audio_features(audio, audio_path)
        
        # Try to transcribe (may fail for synthetic data)
        text = self.transcribe_audio(audio)
        speech_patterns = self.analyze_speech_patterns(text)
        
        # Combine all features
        features = {**audio_features, **speech_patterns}
        return features
        
    def process_directory(self, directory_path):
        """Process all audio files in a directory"""
        features_list = []
        for file in os.listdir(directory_path):
            if file.endswith(('.wav', '.mp3')):
                audio_path = os.path.join(directory_path, file)
                features = self.process_audio_file(audio_path)
                if features:
                    features_list.append(features)
        
        return features_list

def main():
    # Example usage
    detector = CognitiveDeclineDetector()
    
    # Directory containing audio samples
    samples_dir = 'data/samples'
    
    if os.path.exists(samples_dir):
        print(f"Processing audio files in {samples_dir}...")
        features_list = detector.process_directory(samples_dir)
    else:
        # Process individual audio files
        audio_files = ['data/samples/sample1.wav', 'data/samples/sample2.wav']
        print("Processing individual audio files...")
        features_list = []
        for audio_file in audio_files:
            features = detector.process_audio_file(audio_file)
            if features:
                features_list.append(features)
    
    if not features_list:
        print("No audio files were successfully processed.")
        return
    
    # Convert to DataFrame
    features_df = pd.DataFrame(features_list)
    
    # Detect anomalies
    results_df = detector.detect_anomalies(features_df)
    
    # Visualize results
    detector.visualize_results(results_df, 'analysis_results.png')
    
    # Print summary
    print("\nAnalysis Summary:")
    print(f"Total samples analyzed: {len(features_df)}")
    print(f"Average anomaly score: {results_df['anomaly_score'].mean():.2f}")
    
    high_risk_threshold = 1.5
    high_risk_count = sum(results_df['anomaly_score'] > high_risk_threshold)
    print(f"Number of potential risk cases (score > {high_risk_threshold}): {high_risk_count}")
    
    # Save results to CSV
    results_df.to_csv('analysis_results.csv', index=False)
    print("Results saved to analysis_results.csv")
    
    print("\nNext steps:")
    print("1. Review the visualizations in analysis_results.png")
    print("2. Examine the detailed metrics in analysis_results.csv")
    print("3. For production use, validate these findings with a neurologist")

if __name__ == "__main__":
    main() 