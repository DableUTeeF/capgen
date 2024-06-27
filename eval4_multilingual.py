from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    ViTImageProcessor,
    VisionEncoderDecoderModel,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
)
from PIL import Image
import numpy as np
from matplotlib import pyplot as plt
from transformers.modeling_outputs import BaseModelOutput, Seq2SeqLMOutput
import torch
import os
import json


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
        embeded_instruction = self.decoder.transformer.wte(instruction_tokens)
    elif hasattr(self.decoder, 'roberta'):
        embeded_instruction = self.decoder.roberta.embeddings(instruction_tokens)
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
    inputs = model_inputs.pop(model.main_input_name)
    model_outputs = model.generate(inputs, **model_inputs, **generate_kwargs)
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


def generate(inputs):
    model_inputs = preprocess(inputs, image_processor).to('cuda')
    model_outputs = fwd(model, model_inputs)
    outputs = postprocess(model_outputs)
    return outputs


VisionEncoderDecoderModel.forward = forward
if __name__ == '__main__':
    cp = '/mnt/c/Users/Admin/Downloads/work/vqa/cp/gpt2-baseline_8/checkpoint-925000'
    model = VisionEncoderDecoderModel.from_pretrained(cp).eval().cuda()
    tokenizer = AutoTokenizer.from_pretrained(cp)
    image_processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224-in21k')
    thai_tokens = tokenizer(['รูปของ'], padding="max_length",
                            max_length=12,
                            return_tensors="pt",
                            truncation=True).input_ids.to('cuda')
    english_tokens = tokenizer(['image of'], padding="max_length",
                               max_length=12,
                               return_tensors="pt",
                               truncation=True).input_ids.to('cuda')

    inputs = '/home/palm/data/coco/images/val2017/000000000285.jpg'
    image = Image.open(inputs)
    # instruction_tokens = english_tokens
    # english_text = generate('/home/palm/data/coco/images/val2017/000000000285.jpg')
    instruction_tokens = tokenizer(
        (
            'Is the bear sitting?',
            'Is it daytime?',
            'Is the bears eyes open?',
        ),
        padding=True,
        # max_length=50,
        return_tensors="pt",
        truncation=True
    ).input_ids.to('cuda')
    thai_text = generate('/home/palm/data/coco/images/val2017/000000000285.jpg')

    src = '/home/palm/data/coco/images/val2017/'
    data = {}
    for file in os.listdir(src):
        text = generate(os.path.join(src, file))
        data[file] = text
    json.dump(data, open('out/wangchan-multi.json', 'w'))
