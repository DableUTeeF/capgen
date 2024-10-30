from transformers import AutoConfig, Blip2ForConditionalGeneration, AutoImageProcessor, ViTModel, Blip2QFormerModel, AutoModelForCausalLM, AutoModelForSeq2SeqLM
from torch import nn
import torch


class Blip2ForConditionalGeneration(Blip2ForConditionalGeneration):
    def __init__(self, config):
        super().__init__(config)

        self.vision_model = ViTModel(config.vision_config)

        self.query_tokens = nn.Parameter(torch.zeros(1, config.num_query_tokens, config.qformer_config.hidden_size))
        self.qformer = Blip2QFormerModel(config.qformer_config)

        self.language_projection = nn.Linear(config.qformer_config.hidden_size, config.text_config.hidden_size)
        if config.use_decoder_only_language_model:
            language_model = AutoModelForCausalLM.from_config(config.text_config)
        else:
            language_model = AutoModelForSeq2SeqLM.from_config(config.text_config)

        # Update _tied_weights_keys using the base model used.
        if language_model._tied_weights_keys is not None:
            self._tied_weights_keys = [f"language_model.{k}" for k in language_model._tied_weights_keys]

        self.language_model = language_model

        # Initialize weights and apply final processing
        self.post_init()


if __name__ == '__main__':
    # Blip2ForConditionalGeneration.__init__ = __init__
    vit_model = 'google/vit-base-patch16-224-in21k'
    blip_model = 'Salesforce/blip2-opt-2.7b'
    text_decode_model = 'gpt2'
    feature_extractor = AutoImageProcessor.from_pretrained(vit_model)
    cfg = AutoConfig.from_pretrained(blip_model)
    lm_cfg = AutoConfig.from_pretrained(text_decode_model)
    lm_cfg.is_decoder = True
    vs_cfg = AutoConfig.from_pretrained(vit_model)
    vs_cfg.attention_dropout = 0
    cfg.qformer_config.hidden_size = 394
    cfg.qformer_config.encoder_hidden_size = 768
    cfg.qformer_config.num_attention_heads = 2
    cfg.qformer_config.num_hidden_layers = 24
    cfg.text_config = lm_cfg
    cfg.vision_config = vs_cfg
    cfg.use_decoder_only_language_model = True
    blimp = Blip2ForConditionalGeneration(cfg)

    print()
