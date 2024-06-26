from transformers import AutoTokenizer, VisionEncoderDecoderModel, CLIPProcessor, AutoModel, AutoModelForCausalLM
from torch import nn
from transformers.modeling_outputs import BaseModelOutput
import torch
from PIL import Image


class Encoder(nn.Module):
    def __init__(self, vision_model, visual_projection, config, size=512):
        super().__init__()
        self.vision_model = vision_model
        self.visual_projection = visual_projection
        self.config = config
        self.config.hidden_size = size

    def forward(self, pixel_values, output_attentions=None, output_hidden_states=None, return_dict=True, **kwargs):
        vision_outputs = self.vision_model(
            pixel_values=pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        pooled_output = vision_outputs[1]  # pooled_output
        image_features = self.visual_projection(pooled_output).view(pooled_output.size(0), 1, -1)
        return BaseModelOutput(last_hidden_state=image_features)

    def get_output_embeddings(self):
        pass


if __name__ == '__main__':
    clip_model = 'openai/clip-vit-base-patch32'
    clip = AutoModel.from_pretrained(clip_model)
    tokenizer = AutoTokenizer.from_pretrained('gpt2')
    feature_extractor = CLIPProcessor.from_pretrained(clip_model)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    encoder = Encoder(
        clip.vision_model,
        clip.visual_projection,
        clip.vision_model.config
    )
    decoder = AutoModelForCausalLM.from_pretrained(
        'gpt2',
        is_decoder=True,
        add_cross_attention=True
    )
    base_model = VisionEncoderDecoderModel(
        None,
        encoder,
        decoder
    )

    base_model.load_state_dict(torch.load('/mnt/d/work/capgen/workdir/clip_multi_freeze_proj_nocross_2_1e-05_encoder_freeze/pytorch_model.bin'))

    start_tokens = tokenizer(
        'รูปของ',
        padding="max_length",
        max_length=12,
        return_tensors="pt",
        truncation=True
    ).input_ids

    encoder_hidden_states = clip.get_text_features(**feature_extractor(text='this is a cat', return_tensors="pt")).view(1, 1, -1)
    encoder_hidden_states = clip.get_image_features(**feature_extractor(images=Image.open('/mnt/c/Users/Admin/Pictures/360px-A_sunflower.jpg'), return_tensors="pt")).view(1, 1, -1)

    encoder_hidden_states = base_model.enc_to_dec_proj(encoder_hidden_states)
    with torch.no_grad():
        generated_tokens = start_tokens
        for i in range(120):
            embed_tokens = base_model.decoder.transformer.wte(generated_tokens)
            hidden_states = torch.cat((encoder_hidden_states, embed_tokens), dim=1)
            decoder_outputs = base_model.decoder(
                inputs_embeds=hidden_states,  # n, 50, 768
                output_attentions=None,
                output_hidden_states=None,
                use_cache=None,
                past_key_values=None,
                return_dict=None,
            )
            logits = decoder_outputs.logits
            token = logits[:, -1:].argmax(-1)
            generated_tokens = torch.cat((generated_tokens, token), dim=1)
            if token == tokenizer.eos_token_id:
                break
    print('an image of a cat', tokenizer.decode(generated_tokens[0]))
