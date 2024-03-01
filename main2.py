from transformers import pipeline
from datasets import load_dataset
import soundfile as sf
import torch
from pydub import AudioSegment
from pydub.playback import play
import os

synthesiser = pipeline("text-to-speech", "microsoft/speecht5_tts")

embeddings_dataset = load_dataset("Matthijs/cmu-arctic-xvectors", split="validation")
speaker_embedding = torch.tensor(embeddings_dataset[7306]["xvector"]).unsqueeze(0)

speech = synthesiser("deep is chutiyo", forward_params={"speaker_embeddings": speaker_embedding})

# Specify the manual file path for the WAV file
manual_wav_file_path = r"C:\Users\PARAM M. SURELIYA\PycharmProjects\text_to_speech\manual_audio1.wav"

# Save the audio to the specified WAV file
sf.write(manual_wav_file_path, speech["audio"], samplerate=speech["sampling_rate"])


