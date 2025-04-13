import os
import numpy as np
import wave
import struct
from scipy.io import wavfile
import json

# Create directory for samples if it doesn't exist
os.makedirs('data/samples', exist_ok=True)

def create_sine_wave(freq, duration, sample_rate=16000):
    """Create a sine wave at the given frequency and duration"""
    t = np.linspace(0, duration, int(sample_rate * duration))
    wave = np.sin(2 * np.pi * freq * t)
    return wave

def add_pauses(wave, num_pauses, pause_duration, sample_rate=16000):
    """Add random pauses to the audio signal"""
    result = wave.copy()
    pause_samples = int(pause_duration * sample_rate)
    for _ in range(num_pauses):
        # Choose random position for pause
        pos = np.random.randint(0, len(result) - pause_samples)
        # Insert pause (silence)
        result[pos:pos+pause_samples] = 0
    return result

def add_hesitations(wave, num_hesitations, hesitation_duration, sample_rate=16000):
    """Add hesitation sounds (low amplitude noise) to the audio signal"""
    result = wave.copy()
    hesitation_samples = int(hesitation_duration * sample_rate)
    for _ in range(num_hesitations):
        # Choose random position for hesitation
        pos = np.random.randint(0, len(result) - hesitation_samples)
        # Create hesitation sound (low amplitude noise)
        hesitation = np.random.normal(0, 0.05, hesitation_samples)
        # Insert hesitation
        result[pos:pos+hesitation_samples] = hesitation
    return result

def save_wave_file(filename, wave_data, sample_rate=16000):
    """Save wave data to a WAV file"""
    # Normalize the data to -1 to 1
    normalized_data = wave_data / np.max(np.abs(wave_data))
    # Convert to 16-bit integers
    scaled_data = (normalized_data * 32767).astype(np.int16)
    # Save to file
    wavfile.write(filename, sample_rate, scaled_data)
    
def generate_sample(filename, freq_base, duration, num_pauses, num_hesitations, cognitive_level):
    """Generate a sample audio file with cognitive decline features"""
    sample_rate = 16000
    # Create base audio (speech simulation)
    wave_data = create_sine_wave(freq_base, duration, sample_rate)
    
    # Add frequency variations to simulate speech
    for i in range(1, 5):
        harmonic = create_sine_wave(freq_base * i * 1.5, duration, sample_rate)
        wave_data += harmonic * 0.5
    
    # Add pauses and hesitations based on cognitive level
    wave_data = add_pauses(wave_data, num_pauses, 0.2, sample_rate)
    wave_data = add_hesitations(wave_data, num_hesitations, 0.15, sample_rate)
    
    # Save to file
    save_wave_file(filename, wave_data, sample_rate)
    
    return {
        'filename': filename,
        'duration': duration,
        'num_pauses': num_pauses,
        'num_hesitations': num_hesitations,
        'cognitive_level': cognitive_level
    }

# Generate samples with varying levels of cognitive decline indicators
samples = []

# Sample 1: Normal speech (few pauses and hesitations)
samples.append(generate_sample('data/samples/sample1.wav', 220, 10, 2, 1, 'normal'))

# Sample 2: Mild cognitive decline (moderate pauses and hesitations)
samples.append(generate_sample('data/samples/sample2.wav', 200, 10, 5, 3, 'mild'))

# Sample 3: Moderate cognitive decline (many pauses and hesitations)
samples.append(generate_sample('data/samples/sample3.wav', 180, 10, 8, 6, 'moderate'))

# Sample 4: Severe cognitive decline (excessive pauses and hesitations)
samples.append(generate_sample('data/samples/sample4.wav', 160, 10, 12, 9, 'severe'))

# Sample 5: Another normal sample (for comparison)
samples.append(generate_sample('data/samples/sample5.wav', 240, 10, 1, 1, 'normal'))

# Save metadata
with open('data/samples/metadata.json', 'w') as f:
    json.dump(samples, f, indent=2)

print(f"Generated {len(samples)} sample audio files in data/samples/")
for sample in samples:
    print(f"- {sample['filename']}: {sample['cognitive_level']} level, {sample['num_pauses']} pauses, {sample['num_hesitations']} hesitations") 