from transformers import (
    AutoModelForCausalLM,
    AutoModel,
    AutoTokenizer,
    AutoImageProcessor,
    AutoProcessor,
    VisionEncoderDecoderModel,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
)
from torch import nn
from transformers.modeling_outputs import BaseModelOutput


class Encoder(nn.Module):
    def __init__(self, vision_model, visual_projection, config, size=512):
        super().__init__()
        self.vision_model = vision_model
        self.visual_projection = visual_projection
        self.config = config
        self.config.hidden_size = size

    def forward(self, pixel_values, output_attentions, output_hidden_states, return_dict, **kwargs):
        vision_outputs = self.vision_model(
            pixel_values=pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        pooled_output = vision_outputs[1]  # pooled_output
        image_features = self.visual_projection(pooled_output)
        return BaseModelOutput(last_hidden_state=image_features)

    def get_output_embeddings(self):
        pass


if __name__ == '__main__':
    vit_model = 'openai/clip-vit-base-patch32'
    text_decode_model = 'gpt2'
    clip = AutoModel.from_pretrained(vit_model)
    encoder = Encoder(
        clip.vision_model,
        clip.visual_projection,
        clip.vision_model.config
    )
    encoder.get_output_embeddings = t
    decoder = AutoModelForCausalLM.from_pretrained(
        text_decode_model,
        is_decoder=True,
        add_cross_attention=True
    )
    base_model = VisionEncoderDecoderModel(
        None,
        encoder,
        decoder
    )
    feature_extractor = AutoProcessor.from_pretrained(vit_model)

