"""
whisper ASR Module
==================

This module provides on-device ASR capabilities by using the whisper transformer
provided by huggingface. In addition, the ASR module provides end-of-utterance detection
(with a VAD), so that produced hypotheses are being "committed" once an utterance is
finished.
"""

import threading
import retico_core
from retico_core.audio import AudioIU
from retico_core.text import SpeechRecognitionIU
from transformers import WhisperForConditionalGeneration, WhisperProcessor
import transformers
import pydub
import webrtcvad
import numpy as np
import time

transformers.logging.set_verbosity_error()

class WhisperASR:
    def __init__(
        self,
        whisper_model="openai/whisper-base",
        framerate=16_000,
        silence_dur=1,
        vad_agressiveness=3,
        silence_threshold=0.75,
    ):
        self.processor = WhisperProcessor.from_pretrained(whisper_model)
        self.model = WhisperForConditionalGeneration.from_pretrained(whisper_model)
        self.model.config.forced_decoder_ids = None

        # force language to french
        # forced_decoder_ids = self.processor.get_decoder_prompt_ids(
        # language="french", task="transcribe")
        
        # translation from french to english
        # forced_decoder_ids = self.processor.get_decoder_prompt_ids(
        # language="french", task="translate")

        self.audio_buffer = []
        self.framerate = framerate
        self.vad = webrtcvad.Vad(vad_agressiveness)
        self.silence_dur = silence_dur
        self.vad_state = False
        self._n_sil_frames = None
        self.silence_threshold = silence_threshold

    def _resample_audio(self, audio):
        if self.framerate != 16_000:
            # resample if framerate is not 16 kHz
            s = pydub.AudioSegment(
            audio, sample_width=2, channels=1, frame_rate=self.framerate
            )
            s = s.set_frame_rate(16_000)
            return s._data
        return audio
        
    def get_n_sil_frames(self):
        if not self._n_sil_frames:
            if len(self.audio_buffer) == 0:
                return None
            frame_length = len(self.audio_buffer[0]) / 2
            self._n_sil_frames = int(self.silence_dur / (frame_length / 16_000))
        return self._n_sil_frames
        
    def recognize_silence(self):
        n_sil_frames = self.get_n_sil_frames()
        if not n_sil_frames or len(self.audio_buffer) < n_sil_frames:
            return True
        silence_counter = 0

        # count the number of frames with silence
        for a in self.audio_buffer[-n_sil_frames:]:
            if not self.vad.is_speech(a, 16_000):
                silence_counter += 1
        
        # if the amount of silence is greater than threshold return True 
        if silence_counter >= int(self.silence_threshold * n_sil_frames):
            return True
        return False
    
    def add_audio(self, audio):
        audio = self._resample_audio(audio)
        self.audio_buffer.append(audio)

    def recognize(self):
        silence = self.recognize_silence()

        if not self.vad_state and not silence:
            self.vad_state = True
            self.audio_buffer = self.audio_buffer[-self.get_n_sil_frames() :]

        if not self.vad_state:
            return None, False

        full_audio = b""
        for a in self.audio_buffer:
            full_audio += a
        npa = np.frombuffer(full_audio, dtype=np.int16).astype(np.double)
        if len(npa) < 10:
            return None, False
        input_values = self.processor(
            npa, return_tensors="pt", sampling_rate=16000
        ).input_values
        logits = self.model(input_values).logits
        predicted_ids = np.argmax(logits.detach().numpy(), axis=-1)
        transcription = self.processor.batch_decode(predicted_ids)[0].lower()

        if silence:
            self.vad_state = False
            self.audio_buffer = []

        return transcription, self.vad_state
            

    def reset(self):
        self.vad_state = True
        self.audio_buffer = []    

class WhisperASRModule(retico_core.AbstractModule):
    @staticmethod
    def name():
        return "Whipser ASR Module"
    
    @staticmethod
    def description():
        return "A module that recognizes speech using Whisper."
    
    @staticmethod
    def input_ius():
        return [AudioIU]

    @staticmethod
    def output_iu():
        return SpeechRecognitionIU