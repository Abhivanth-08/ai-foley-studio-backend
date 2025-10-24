import numpy as np
import librosa

def extract_second_audio_librosa(file_path, target_second=0, sample_rate=22050):
    try:
        # Load audio file
        audio_data, sr = librosa.load(file_path, sr=sample_rate)

        # Calculate start and end samples for the target second
        start_sample = target_second * sr
        end_sample = (target_second + 1) * sr

        # Ensure we don't go beyond the audio length
        if start_sample >= len(audio_data):
            raise ValueError(f"Target second {target_second} is beyond audio length")

        end_sample = min(end_sample, len(audio_data))

        # Extract the second
        second_audio = audio_data[start_sample:end_sample]

        # If the audio is shorter than 1 second, pad with zeros
        if len(second_audio) < sr:
            second_audio = np.pad(second_audio, (0, sr - len(second_audio)))

        return second_audio, sr

    except Exception as e:
        print(f"Error processing audio: {e}")
        return None, None

