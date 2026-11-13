# retico-whisperasr
A local Whisper ASR module for ReTiCo. It transcribes speech with OpenAI's [Whisper](https://github.com/openai/whisper) model (`openai/whisper-base` from Hugging Face) and detects the end of an utterance with a voice activity detector (VAD), so that its hypotheses are committed once the user stops speaking. See the citation below for model information.

## Installation and Requirements

To use the Whisper ASR module you first need to install the retico-whisperasr package:
* Install retico-whisperasr:
```pip install retico-whisperasr```

  or, for the latest development version from GitHub:
```pip install git+https://github.com/retico-team/retico-whisperasr.git```

## Modules

### `WhisperASRModule`
Subscribes to `AudioIU` and outputs `SpeechRecognitionIU`. While the user is speaking, the module transcribes the audio about every 0.5 seconds and sends the new words as they are recognized. Once the VAD detects `silence_dur` seconds of silence, the words of the utterance are committed.

#### Arguments:
* `framerate` (int): the sample rate of the incoming audio; audio that isn't 16 kHz is resampled automatically; defaults to `None` (uses the rate of the incoming `AudioIU`)
* `silence_dur` (float): seconds of silence after which the utterance is considered finished and committed; defaults to `1`
* `language` (str): the language spoken, e.g. `"german"`; defaults to `None` (English)
* `task` (str): `"transcribe"` to transcribe in the spoken language, or `"translate"` to translate into English; only used when `language` is set; defaults to `"transcribe"`

### Example
**Note:** The example uses `MicrophoneModule` and `TextPrinterModule` from `retico_core`.

```python
import retico_core
from retico_core.audio import MicrophoneModule
from retico_core.debug import TextPrinterModule
from retico_whisperasr import WhisperASRModule

microphone = MicrophoneModule(rate=16000)
asr = WhisperASRModule()
printer = TextPrinterModule()

microphone.subscribe(asr)
asr.subscribe(printer)

retico_core.network.run(microphone)

print("Network is running")
input()

retico_core.network.stop(microphone)
```

Citation
```
@misc{radford2022whisper,
  doi = {10.48550/ARXIV.2212.04356},
  url = {https://arxiv.org/abs/2212.04356},
  author = {Radford, Alec and Kim, Jong Wook and Xu, Tao and Brockman, Greg and McLeavey, Christine and Sutskever, Ilya},
  title = {Robust Speech Recognition via Large-Scale Weak Supervision},
  publisher = {arXiv},
  year = {2022},
  copyright = {arXiv.org perpetual, non-exclusive license}
}
```
