import torch
import torch_directml
import torchaudio
import torchvision.transforms as T
from Model import UNet
import os
import math
import numpy as np

# --- SETTINGS ---
MODEL_PATH = "best_model.pth"
INPUT_AUDIO_PATH = r"C:\Users\babri\Music\1-01 Piano Man.wav" 
OUTPUT_FOLDER = "stereo_threshold_results"
CHUNK_DURATION = 5 

# --- THE MAGIC NUMBER ---
# Any pixel in the mask below this value becomes pure black (0).
# Try 0.2, 0.3, or 0.4. 
# Higher = Cleaner silence, but might cut off quiet notes.
MASK_THRESHOLD = 0.1


import os
os.environ["PATH"] += os.pathsep + os.getcwd()

DEVICE = torch_directml.device() if torch_directml.is_available() else "cpu"

def separate_full_song(model_path, audio_path, output_dir):
    print(f"Loading STEREO model from {model_path}...")
    
    model = UNet(num_classes=8).to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE, weights_only=False))
    model.eval()

    print(f"Loading song: {audio_path}")
    waveform, sample_rate = torchaudio.load(audio_path)
    
    total_duration_sec = waveform.shape[1] / sample_rate
    num_chunks = math.ceil(total_duration_sec / CHUNK_DURATION)
    print(f"Processing with Threshold: {MASK_THRESHOLD}")

    stems_storage = [[], [], [], []] 
    stem_names = ['vocals', 'drums', 'bass', 'other']

    n_fft = 2048
    hop_length = 512
    window = torch.hann_window(n_fft) 
    resize_transform = T.Resize((512, 512), antialias=True)
    samples_per_chunk = CHUNK_DURATION * sample_rate
    
    for i in range(num_chunks):
        start = i * samples_per_chunk
        end = min((i + 1) * samples_per_chunk, waveform.shape[1])
        
        chunk_waveform = waveform[:, start:end]
        original_length = chunk_waveform.shape[1]
        
        if original_length < samples_per_chunk:
            padding = samples_per_chunk - original_length
            chunk_waveform = torch.nn.functional.pad(chunk_waveform, (0, padding))

        # 1. STFT
        complex_spec = torch.stft(chunk_waveform, n_fft=n_fft, hop_length=hop_length, 
                                window=window, return_complex=True)
        magnitude = torch.abs(complex_spec)
        phase = torch.angle(complex_spec)

        # 2. Inference
        model_input = resize_transform(magnitude).unsqueeze(0)
        model_input = torch.log1p(model_input).to(DEVICE)
        
        with torch.no_grad():
            predicted_masks = model(model_input) 
        
        predicted_masks = predicted_masks.squeeze(0).cpu()
        
        # 3. Restore Size
        original_spec_size = magnitude.shape[-2:] 
        restore_transform = T.Resize(original_spec_size, antialias=True)
        restored_masks = restore_transform(predicted_masks)

        # --- STEP 4: APPLY THRESHOLD (The Fix) ---
        # "If confidence is low, kill it."
        restored_masks[restored_masks < MASK_THRESHOLD] = 0.0

        # --- STEP 5: RESIDUAL SUBTRACTION ---
        
        mask_drums = restored_masks[2:4]
        mask_bass  = restored_masks[4:6]
        mask_other = restored_masks[6:8]
        
        # Calculate Accompaniment
        spec_drums = (magnitude * mask_drums) * torch.exp(1j * phase)
        spec_bass  = (magnitude * mask_bass)  * torch.exp(1j * phase)
        spec_other = (magnitude * mask_other) * torch.exp(1j * phase)
        
        # Calculate Vocals via Subtraction
        # Since we cleaned up the masks above, the subtraction will be sharper!
        original_mix_complex = magnitude * torch.exp(1j * phase)
        spec_vocals = original_mix_complex - (spec_drums + spec_bass + spec_other)
        
        def istft_convert(complex_t):
            return torch.istft(complex_t, n_fft=n_fft, hop_length=hop_length, 
                               window=window, length=chunk_waveform.shape[1])

        wav_vocals = istft_convert(spec_vocals)
        wav_drums  = istft_convert(spec_drums)
        wav_bass   = istft_convert(spec_bass)
        wav_other  = istft_convert(spec_other)
        
        if original_length < samples_per_chunk:
            wav_vocals = wav_vocals[:, :original_length]
            wav_drums  = wav_drums[:, :original_length]
            wav_bass   = wav_bass[:, :original_length]
            wav_other  = wav_other[:, :original_length]

        stems_storage[0].append(wav_vocals)
        stems_storage[1].append(wav_drums)
        stems_storage[2].append(wav_bass)
        stems_storage[3].append(wav_other)
            
        print(f"Processed chunk {i+1}/{num_chunks}", end='\r')

    print("\nStitching and Saving...")
    os.makedirs(output_dir, exist_ok=True)
    
    for i, name in enumerate(stem_names):
        full_stem = torch.cat(stems_storage[i], dim=1).cpu()
        if full_stem.shape[1] < waveform.shape[1]:
            diff = waveform.shape[1] - full_stem.shape[1]
            full_stem = torch.nn.functional.pad(full_stem, (0, diff))
        elif full_stem.shape[1] > waveform.shape[1]:
            full_stem = full_stem[:, :waveform.shape[1]]
            
        torchaudio.save(os.path.join(output_dir, f"{name}.wav"), full_stem, sample_rate)

    # Reconstruct Accompaniment properly from the thresholded stems
    acc = stems_storage[1][0] # Dummy
    vocals_wav, _ = torchaudio.load(os.path.join(output_dir, "vocals.wav"))
    acc_waveform = waveform - vocals_wav
    torchaudio.save(os.path.join(output_dir, "accompaniment.wav"), acc_waveform, sample_rate)
    print("Done!")

if __name__ == '__main__':
    separate_full_song(MODEL_PATH, INPUT_AUDIO_PATH, OUTPUT_FOLDER)