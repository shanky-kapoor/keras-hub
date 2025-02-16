from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.models.causal_lm_preprocessor import CausalLMPreprocessor
from keras_hub.src.models.whisper.whisper_backbone import WhisperBackbone
from keras_hub.src.models.whisper.whisper_tokenizer import WhisperTokenizer

@keras_hub_export("keras_hub.models.WhisperCausalLMPreprocessor")
class WhisperCausalLMPreprocessor(CausalLMPreprocessor):
    """Whisper Causal Language Model Preprocessor."""

    backbone_cls = WhisperBackbone
    tokenizer_cls = WhisperTokenizer