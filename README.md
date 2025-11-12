# retico-whisperasr
Local whisper ASR Module for ReTiCo. See citation below for model information. 

### Installations and requirements

```
pip install git+https://github.com/retico-team/retico-whisperasr.git

```


### Example

```python
import sys
from retico import *

prefix = '/path/to/module/'
sys.path.append(prefix+'retico-whisperasr')

from retico_whisperasr import WhisperASRModule



microphone = modules.MicrophoneModule(rate=16000)

asr = WhisperASRModule()

printer = modules.TextPrinterModule()
debug = modules.DebugModule()

microphone.subscribe(asr)
asr.subscribe(printer)
# asr.subscribe(debug)

run(microphone)

print("Network is running")
input()

stop(microphone)
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