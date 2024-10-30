from transformers import Blip2ForConditionalGeneration, AutoTokenizer, AutoModel, AutoImageProcessor, AutoModelForCausalLM, Blip2Config, VisionEncoderDecoderModel
from torch import nn
import torch
from PIL import Image
from calflops import calculate_flops, calculate_flops_hf
from transformers.modeling_outputs import BaseModelOutput
import pandas as pd
from calflops.calculate_pipline import CalFlopsPipline


class Encoder(nn.Module):
    def __init__(self, vision_model, visual_projection, config, size=512):
        super().__init__()
        self.vision_model = vision_model
        self.visual_projection = visual_projection
        self.config = config
        self.config.hidden_size = size

    def forward(self, pixel_values, output_attentions=None, output_hidden_states=None, return_dict=False, **kwargs):
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


def cat_cost(model):
    calculate_flops_pipline = CalFlopsPipline(model=model,
                                              include_backPropagation=False,
                                              compute_bp_factor=2.0)
    calculate_flops_pipline.start_flops_calculate(ignore_list=None)

    model_inputs = image_processor(image, return_tensors='pt').to('cuda')
    encoder_hidden_states = model.encoder(
        model_inputs['pixel_values'],
    )[0]
    if len(encoder_hidden_states.size()) == 4:
        encoder_hidden_states = encoder_hidden_states.permute(0, 2, 3, 1)
        encoder_hidden_states = encoder_hidden_states.reshape(encoder_hidden_states.size(0), -1, encoder_hidden_states.size(3))
    encoder_hidden_states = model.enc_to_dec_proj(encoder_hidden_states)
    generated_tokens = torch.tensor([[
        tokenizer.bos_token_id
    ]]).cuda()
    generated_tokens.size()
    if 'GPT' in model.decoder.config.architectures[0]:
        hidden_states = encoder_hidden_states
        decoder_outputs = model.decoder(
            inputs_embeds=hidden_states,  # n, 50, 768
            output_attentions=None,
            output_hidden_states=None,
            use_cache=None,
            past_key_values=None,
            return_dict=None,
        )
        logits = decoder_outputs.logits
        token = logits[:, -1:].argmax(-1)
        generated_tokens = token
    for i in range(119):
        if 'GPT' in model.decoder.config.architectures[0]:
            embed_tokens = model.decoder.transformer.wte(generated_tokens)
        elif 'Camembert' in model.decoder.config.architectures[0]:
            embed_tokens = model.decoder.roberta.embeddings(generated_tokens)
        elif 'Roberta' in model.decoder.config.architectures[0]:
            embed_tokens = model.decoder.roberta.embeddings(generated_tokens)
        elif 'OPT' in model.decoder.config.architectures[0]:
            embed_tokens = model.decoder.model.decoder.embed_tokens(generated_tokens)

        hidden_states = torch.cat((encoder_hidden_states, embed_tokens), dim=1)
        decoder_outputs = model.decoder(
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
    flops = calculate_flops_pipline.get_total_flops()
    macs = calculate_flops_pipline.get_total_macs()
    params = calculate_flops_pipline.get_total_params()
    calculate_flops_pipline.end_flops_calculate()
    return flops / 1e9, macs / 1e9, params


def cross_cost(model):
    calculate_flops_pipline = CalFlopsPipline(model=model,
                                              include_backPropagation=False,
                                              compute_bp_factor=2.0)
    calculate_flops_pipline.start_flops_calculate(ignore_list=None)

    model_inputs = image_processor(image, return_tensors='pt').to('cuda')
    encoder_hidden_states = model.encoder(
        model_inputs['pixel_values'],
    )[0]
    if len(encoder_hidden_states.size()) == 4:
        encoder_hidden_states = encoder_hidden_states.permute(0, 2, 3, 1)
        encoder_hidden_states = encoder_hidden_states.reshape(encoder_hidden_states.size(0), -1, encoder_hidden_states.size(3))
    encoder_hidden_states = model.enc_to_dec_proj(encoder_hidden_states)
    generated_tokens = torch.tensor([[
        tokenizer.bos_token_id
    ]]).cuda()
    generated_tokens.size()
    for i in range(120):

        decoder_outputs = model.decoder(
            input_ids=generated_tokens,
            encoder_hidden_states=encoder_hidden_states,
        )
        logits = decoder_outputs.logits
        token = logits[:, -1:].argmax(-1)
        generated_tokens = torch.cat((generated_tokens, token), dim=1)
    flops = calculate_flops_pipline.get_total_flops()
    macs = calculate_flops_pipline.get_total_macs()
    params = calculate_flops_pipline.get_total_params()
    calculate_flops_pipline.end_flops_calculate()
    return flops / 1e9, macs / 1e9, params


if __name__ == '__main__':
    final = {'type': [], 'image': [], 'text': [], 'flops': [], 'macs': [], 'params': []}
    encoders = [
        'vit',
        # 'swin',
        # 'cnvnext',
        'clip',
        # 'blip2',
    ]
    decoders = [
        'gpt2',
        'wangchan',
        # 'phayathai',
        # 'opt2'
    ]
    image = Image.open('/home/palm/Pictures/52104916_255240975405106_8237557676691685376_n.jpg')
    for vit_model in encoders:
        for text_decode_model in decoders:
            if vit_model == 'clip':
                clip = AutoModel.from_pretrained('openai/clip-vit-base-patch32')
                encoder = Encoder(
                    clip.vision_model,
                    clip.visual_projection,
                    clip.vision_model.config
                )
                image_processor = AutoImageProcessor.from_pretrained('openai/clip-vit-base-patch32')
            elif vit_model == 'swin':
                encoder = AutoModel.from_pretrained('microsoft/swinv2-large-patch4-window12to16-192to256-22kto1k-ft')
                image_processor = AutoImageProcessor.from_pretrained('microsoft/swinv2-large-patch4-window12to16-192to256-22kto1k-ft')
            elif vit_model == 'convnext':
                encoder = AutoModel.from_pretrained('facebook/convnextv2-base-22k-224')
                image_processor = AutoImageProcessor.from_pretrained('facebook/convnextv2-base-22k-224')
            elif vit_model == 'vit':
                encoder = AutoModel.from_pretrained('google/vit-base-patch16-224-in21k')
                image_processor = AutoImageProcessor.from_pretrained('google/vit-base-patch16-224-in21k')
            elif vit_model == 'blip2':
                blip_ori = Blip2ForConditionalGeneration._from_config(
                    Blip2Config.from_pretrained('Salesforce/blip2-opt-2.7b-coco')
                )
                encoder = blip_ori.vision_model
                image_processor = AutoImageProcessor.from_pretrained('Salesforce/blip2-opt-2.7b-coco')
            if text_decode_model == 'gpt2':
                decoder = AutoModelForCausalLM.from_pretrained(
                    'gpt2',
                    is_decoder=True,
                    add_cross_attention=True
                )
                tokenizer = AutoTokenizer.from_pretrained('gpt2')
            elif text_decode_model == 'wangchan':
                decoder = AutoModelForCausalLM.from_pretrained(
                    'airesearch/wangchanberta-base-att-spm-uncased',
                    is_decoder=True,
                    add_cross_attention=True
                )
                tokenizer = AutoTokenizer.from_pretrained('airesearch/wangchanberta-base-att-spm-uncased')
            elif text_decode_model == 'phayathai':
                decoder = AutoModelForCausalLM.from_pretrained(
                    'clicknext/phayathaibert',
                    is_decoder=True,
                    add_cross_attention=True
                )
                tokenizer = AutoTokenizer.from_pretrained('clicknext/phayathaibert')
            elif text_decode_model == 'opt2':
                blip_ori = Blip2ForConditionalGeneration._from_config(
                    Blip2Config.from_pretrained('Salesforce/blip2-opt-2.7b-coco')
                )
                decoder = blip_ori.language_model
                tokenizer = AutoTokenizer.from_pretrained('Salesforce/blip2-opt-2.7b-coco')
            model = VisionEncoderDecoderModel(
                None,
                encoder,
                decoder
            )
            model.enc_to_dec_proj = nn.Linear(model.encoder.config.hidden_size, model.decoder.config.hidden_size)
            model = model.cuda()

            gflops, gmacs, params = cat_cost(model)
            final['type'].append('cat')
            final['image'].append(vit_model)
            final['text'].append(text_decode_model)
            final['flops'].append(gflops)
            final['macs'].append(gmacs)
            final['params'].append(params)

            gflops, gmacs, params = cross_cost(model)
            final['type'].append('cross')
            final['image'].append(vit_model)
            final['text'].append(text_decode_model)
            final['flops'].append(gflops)
            final['macs'].append(gmacs)
            final['params'].append(params)

    df = pd.DataFrame(final)
    df.sort_values('type').to_csv('csvs/cost.csv', index=False)
