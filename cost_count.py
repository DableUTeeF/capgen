from transformers import Blip2ForConditionalGeneration, AutoTokenizer, AutoModel, AutoImageProcessor, AutoModelForCausalLM, Blip2Config
from torch import nn
import torch
from PIL import Image
from calflops import calculate_flops, calculate_flops_hf
from transformers.modeling_outputs import BaseModelOutput
import pandas as pd


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


if __name__ == '__main__':
    final = {'image': [], 'text': [], 'flops': [], 'macs': [], 'params': []}
    encoders = [
        'vit',
        'swin',
        'cnvnext',
        'clip',
        # 'blip2',
    ]
    decoders = [
        'gpt2',
        'wangchan',
        'phayathai',
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
                    add_cross_attention=False
                )
                tokenizer = AutoTokenizer.from_pretrained('gpt2')
            elif text_decode_model == 'wangchan':
                decoder = AutoModelForCausalLM.from_pretrained(
                    'airesearch/wangchanberta-base-att-spm-uncased',
                    is_decoder=True,
                    add_cross_attention=False
                )
                tokenizer = AutoTokenizer.from_pretrained('airesearch/wangchanberta-base-att-spm-uncased')
            elif text_decode_model == 'phayathai':
                decoder = AutoModelForCausalLM.from_pretrained(
                    'clicknext/phayathaibert',
                    is_decoder=True,
                    add_cross_attention=False
                )
                tokenizer = AutoTokenizer.from_pretrained('clicknext/phayathaibert')
            elif text_decode_model == 'opt2':
                blip_ori = Blip2ForConditionalGeneration._from_config(
                    Blip2Config.from_pretrained('Salesforce/blip2-opt-2.7b-coco')
                )
                decoder = blip_ori.language_model
                tokenizer = AutoTokenizer.from_pretrained('Salesforce/blip2-opt-2.7b-coco')
            proj = nn.Linear(encoder.config.hidden_size, decoder.config.hidden_size)

            inputs = image_processor(image, return_tensors='pt')
            with torch.no_grad():
                outputs = encoder(**inputs)

            ec = calculate_flops(encoder, input_shape=tuple(inputs['pixel_values'].size()), output_as_string=False, print_results=False)
            pc = calculate_flops(proj, input_shape=tuple(outputs[0].size()), output_as_string=False, print_results=False)
            dc = calculate_flops_hf(
                'gpt2',
                decoder,
                forward_mode='generate',
                output_as_string=False,
                print_results=False
            )

            final['image'].append(vit_model)
            final['text'].append(text_decode_model)
            final['flops'].append((ec[0] + pc[0] + dc[0] * 1) / 1e9)
            final['macs'].append((ec[1] + pc[1] + dc[1] * 1) / 1e9)
            final['params'].append((ec[2] + pc[2] + dc[2]))

            print('flops =', (ec[0] + pc[0] + dc[0] * 1) / 1e9)
            print('macs =', (ec[1] + pc[1] + dc[1] * 1) / 1e9)
            print('params =', (ec[2] + pc[2] + dc[2]))
    pd.DataFrame(final).to_csv('csvs/cost.csv', index=False)
