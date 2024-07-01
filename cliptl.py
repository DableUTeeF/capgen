from transformers import AutoTokenizer, VisionEncoderDecoderModel, CLIPProcessor, AutoModel, AutoModelForCausalLM
from torch import nn
from transformers.modeling_outputs import BaseModelOutput, Seq2SeqLMOutput
import torch
from PIL import Image


class Encoder(nn.Module):
    main_input_name = 'pixel_values'

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


def forward(
        self,
        pixel_values=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        encoder_outputs=None,
        past_key_values=None,
        decoder_inputs_embeds=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        **kwargs,
):
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    kwargs_encoder = {argument: value for argument, value in kwargs.items() if not argument.startswith("decoder_")}

    kwargs_decoder = {
        argument[len("decoder_"):]: value for argument, value in kwargs.items() if argument.startswith("decoder_")
    }
    if hasattr(self.decoder, 'transformer'):
        embeded_instruction = self.decoder.transformer.wte(start_tokens)
    elif hasattr(self.decoder, 'roberta'):
        embeded_instruction = self.decoder.roberta.embeddings(start_tokens)
    if encoder_outputs is None:
        if pixel_values is None:
            raise ValueError("You have to specify pixel_values")

        encoder_outputs = self.encoder(
            pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            **kwargs_encoder,
        )
    elif isinstance(encoder_outputs, tuple):
        encoder_outputs = BaseModelOutput(*encoder_outputs)

    encoder_hidden_states = encoder_outputs[0]
    encoder_hidden_states = torch.cat((encoder_hidden_states, embeded_instruction), 1)

    # optionally project encoder_hidden_states
    if (
            self.encoder.config.hidden_size != self.decoder.config.hidden_size
            and self.decoder.config.cross_attention_hidden_size is None
    ):
        encoder_hidden_states = self.enc_to_dec_proj(encoder_hidden_states)

    # else:
    encoder_attention_mask = None

    # Decode
    decoder_outputs = self.decoder(
        input_ids=decoder_input_ids,
        attention_mask=decoder_attention_mask,
        encoder_hidden_states=encoder_hidden_states,  # n, 50, 768
        encoder_attention_mask=encoder_attention_mask,
        inputs_embeds=decoder_inputs_embeds,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        use_cache=use_cache,
        past_key_values=past_key_values,
        return_dict=return_dict,
        **kwargs_decoder,
    )

    if not return_dict:
        return decoder_outputs + encoder_outputs

    return Seq2SeqLMOutput(
        logits=decoder_outputs.logits,
        past_key_values=decoder_outputs.past_key_values,
        decoder_hidden_states=decoder_outputs.hidden_states,
        decoder_attentions=decoder_outputs.attentions,
        cross_attentions=decoder_outputs.cross_attentions,
        encoder_last_hidden_state=encoder_outputs.last_hidden_state,
        encoder_hidden_states=encoder_outputs.hidden_states,
        encoder_attentions=encoder_outputs.attentions,
    )


def preprocess(image, image_processor):
    image = Image.open(image).convert('RGB')

    model_inputs = image_processor(images=image, return_tensors='pt')

    return model_inputs


def fwd(model, model_inputs, generate_kwargs=None):
    if generate_kwargs is None:
        generate_kwargs = {'max_new_tokens': 120}
    model_outputs = model.generate(**model_inputs, **generate_kwargs)
    return model_outputs


def postprocess(model_outputs):
    records = []
    for output_ids in model_outputs:
        record = {
            "generated_text": tokenizer.decode(
                output_ids,
                skip_special_tokens=True,
            )
        }
        records.append(record)
    return records


def generate(model_inputs):
    model_outputs = fwd(base_model, model_inputs)
    outputs = postprocess(model_outputs)
    return outputs


if __name__ == '__main__':
    clip_model = 'openai/clip-vit-base-patch32'
    clip = AutoModel.from_pretrained(clip_model).to('cuda')
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
    ).to('cuda')

    base_model.load_state_dict(torch.load('/mnt/d/work/capgen/workdir/clip_multi_freeze_proj_nocross_2_1e-05_encoder_freeze/pytorch_model.bin'))

    start_tokens = tokenizer(
        'รูปของ',
        padding="max_length",
        max_length=12,
        return_tensors="pt",
        truncation=True
    ).input_ids.to('cuda')

    encoder_hidden_states = clip.get_text_features(**feature_extractor(text='this is a cat', return_tensors="pt").to('cuda')).view(1, 1, -1)

    model_inputs = {'encoder_outputs': BaseModelOutput(last_hidden_state=encoder_hidden_states.cuda())}
    generated_text = generate(model_inputs)
    print(generated_text)

    encoder_hidden_states = clip.get_image_features(**feature_extractor(images=Image.open('/mnt/c/Users/Admin/Pictures/00000.jpg'), return_tensors="pt").to('cuda')).view(1, 1, -1)
    model_inputs = {'encoder_outputs': BaseModelOutput(last_hidden_state=encoder_hidden_states.cuda())}
    generated_text = generate(model_inputs)
    print(generated_text)
