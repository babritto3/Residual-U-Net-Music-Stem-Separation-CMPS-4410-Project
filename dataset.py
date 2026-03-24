import torch
import musdb
import torchaudio
import random
from torch.utils.data import Dataset
import torchvision.transforms as T
import numpy as np

class MusDB_Dataset(Dataset):
    def __init__(self, root_dir, chunk_duration=5, is_train=True, n_fft=2048, hop_length=512):
        super(MusDB_Dataset, self).__init__()
        
        self.chunk_duration = chunk_duration
        self.is_train = is_train
        
        # Load the DB
        db = musdb.DB(root=root_dir)
        subset_to_load = 'train' if is_train else 'test'
        self.mus = db.load_mus_tracks(subsets=subset_to_load)
        
        # Power=1.0 (Magnitude)
        self.spectrogram_transform = torchaudio.transforms.Spectrogram(
            n_fft=n_fft,
            hop_length=hop_length,
            power=1.0 
        )
        
        # High Res Transform (512x512)
        self.resize_transform = T.Resize((512, 512), antialias=True) 

    def __len__(self):
        return len(self.mus)

    def __getitem__(self, index):
        track = self.mus[index]
        track.chunk_duration = self.chunk_duration
        
        # Safety: Ensure track is long enough
        if track.duration < self.chunk_duration:
            track.chunk_start = 0
        else:
            track.chunk_start = random.uniform(0, track.duration - self.chunk_duration)
        
        # Load Audio (Stereo is preserved! Shape: 2, Samples)
        mixture_audio = track.audio.T # Transpose to (2, N)
        vocals_audio = track.targets['vocals'].audio.T
        drums_audio = track.targets['drums'].audio.T
        bass_audio = track.targets['bass'].audio.T
        other_audio = track.targets['other'].audio.T
        
        # --- DATA AUGMENTATION ---
        if self.is_train:
            # 1. Random Gain
            gain = random.uniform(0.5, 1.5)
            mixture_audio = mixture_audio * gain
            vocals_audio = vocals_audio * gain
            drums_audio = drums_audio * gain
            bass_audio = bass_audio * gain
            other_audio = other_audio * gain

            # 2. Random Channel Swap (Flip Left/Right speakers)
            if random.random() < 0.5:
                mixture_audio = np.flip(mixture_audio, axis=0).copy()
                vocals_audio = np.flip(vocals_audio, axis=0).copy()
                drums_audio = np.flip(drums_audio, axis=0).copy()
                bass_audio = np.flip(bass_audio, axis=0).copy()
                other_audio = np.flip(other_audio, axis=0).copy()
        # --------------------------
        
        # Convert to Tensor (Keep 2 Channels)
        mixture_tensor = torch.tensor(mixture_audio, dtype=torch.float32)
        vocals_tensor = torch.tensor(vocals_audio, dtype=torch.float32)
        drums_tensor = torch.tensor(drums_audio, dtype=torch.float32)
        bass_tensor = torch.tensor(bass_audio, dtype=torch.float32)
        other_tensor = torch.tensor(other_audio, dtype=torch.float32)

        # 1. Create Magnitude Spectrograms -> Output: (2, Freq, Time)
        mixture_spec = self.spectrogram_transform(mixture_tensor)
        vocals_spec = self.spectrogram_transform(vocals_tensor)
        drums_spec = self.spectrogram_transform(drums_tensor)
        bass_spec = self.spectrogram_transform(bass_tensor)
        other_spec = self.spectrogram_transform(other_tensor)

        # 2. Resize to High Res (512x512) - Transforms work on (C, H, W) naturally
        mixture_spec_resized = self.resize_transform(mixture_spec)
        vocals_spec_resized = self.resize_transform(vocals_spec)
        drums_spec_resized = self.resize_transform(drums_spec)
        bass_spec_resized = self.resize_transform(bass_spec)
        other_spec_resized = self.resize_transform(other_spec)

        # 3. Log-Scale the INPUT
        mixture_input = torch.log1p(mixture_spec_resized)

        # 4. Create Masks (Stereo!)
        # We calculate separate masks for Left and Right channels
        vocal_mask = vocals_spec_resized / (mixture_spec_resized + 1e-8)
        drum_mask = drums_spec_resized / (mixture_spec_resized + 1e-8)
        bass_mask = bass_spec_resized / (mixture_spec_resized + 1e-8)
        other_mask = other_spec_resized / (mixture_spec_resized + 1e-8)
        
        # Stack them into 8 channels: 
        # [Voc_L, Voc_R, Drum_L, Drum_R, Bass_L, Bass_R, Other_L, Other_R]
        target_masks = torch.cat([
            vocal_mask, 
            drum_mask, 
            bass_mask, 
            other_mask
        ], dim=0)

        target_masks = torch.clamp(target_masks, min=0.0, max=1.0)
        
        return mixture_input, target_masks