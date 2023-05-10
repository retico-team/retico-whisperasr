"""
wav2vec ASR Module
==================

This module provides on-device ASR capabilities by using the wav2vec2 transformer
provided by huggingface. In addition, the ASR module provides end-of-utterance detection
(with a VAD), so that produced hypotheses are being "committed" once an utterance is
finished.
"""
