from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    ViTImageProcessor,
    VisionEncoderDecoderModel,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
)
from PIL import Image
import json
from matplotlib import pyplot as plt
import os
import torch
from transformers.modeling_outputs import BaseModelOutput, Seq2SeqLMOutput


def forward(
        self,
        pixel_values = None,
        decoder_input_ids = None,
        decoder_attention_mask = None,
        encoder_outputs=None,
        past_key_values=None,
        decoder_inputs_embeds = None,
        labels = None,
        use_cache = None,
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
    embed_labels = self.decoder.roberta.embeddings(bos)

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

    encoder_hidden_states = encoder_outputs[0]

    # optionally project encoder_hidden_states
    if (
            self.encoder.config.hidden_size != self.decoder.config.hidden_size
            and self.decoder.config.cross_attention_hidden_size is None
    ):
        encoder_hidden_states = self.enc_to_dec_proj(encoder_hidden_states)
    encoder_hidden_states = torch.cat((encoder_hidden_states, embed_labels), dim=1)

    # Decode
    decoder_outputs = self.decoder(
        inputs_embeds=encoder_hidden_states,  # n, 50, 768
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
        loss=None,
        logits=decoder_outputs.logits,
        past_key_values=decoder_outputs.past_key_values,
        decoder_hidden_states=decoder_outputs.hidden_states,
        decoder_attentions=decoder_outputs.attentions,
        # cross_attentions=decoder_outputs.cross_attentions,
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


@torch.no_grad()
def generate(inputs):
    model_inputs = preprocess(inputs, image_processor).to('cuda')
    model_outputs = fwd(model, model_inputs)
    outputs = postprocess(model_outputs)
    return outputs


if __name__ == '__main__':
    # VisionEncoderDecoderModel.forward = forward
    image_processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224-in21k')
    cp_path = '/mnt/c/Users/Admin/Downloads/work/capgen/cp/'
    for p in [
        'phayathai_vit_nocross_8',
        # 'gpt2_vit_freezevit_8',
        'phayathai_vit_nocross_freezevit_8',
    ]:
        # if 'multi' in p:
        #     continue
        # if p+'.json' in os.listdir('out/tt'):
        #     continue
        cp = os.path.join(cp_path, p)
        model = VisionEncoderDecoderModel.from_pretrained(cp).eval().cuda()
        tokenizer = AutoTokenizer.from_pretrained(cp)
        bos = torch.tensor([[tokenizer.bos_token_id]]).cuda()
        inputs = '/home/palm/data/coco/images/val2017/000000000285.jpg'
        image = Image.open(inputs)
        text = generate('/home/palm/data/coco/images/val2017/000000000285.jpg')

        src = '/home/palm/data/coco/images/val2017/'
        data = {}
        for file in os.listdir(src):
            tokenizer.padding_side = "right"  # Fix weird overflow issue with fp16 training
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.padding_side = "right"  # Fix weird overflow issue with fp16 training

            model.config.eos_token_id = tokenizer.eos_token_id
            model.config.decoder_start_token_id = tokenizer.bos_token_id
            model.config.pad_token_id = tokenizer.pad_token_id
            image = Image.open(os.path.join(src, file)).convert('RGB')
            encoder_inputs = image_processor(images=image, return_tensors="pt").pixel_values
            encoder_outputs = model.encoder(
                encoder_inputs.cuda(),
                output_attentions=None,
                output_hidden_states=None,
                return_dict=None,
            )
            encoder_hidden_states = encoder_outputs[0]
            # encoder_hidden_states = model.enc_to_dec_proj(encoder_hidden_states)
            decoder_outputs = model.decoder(
                inputs_embeds=encoder_hidden_states,  # n, 50, 768
                output_attentions=None,
                output_hidden_states=None,
                use_cache=None,
                past_key_values=None,
                return_dict=None,
            )
            generated_tokens = torch.tensor([[tokenizer.bos_token_id]]).cuda()
            predicted = []
            generated_tokens.size()
            for i in range(120):
                embed_tokens = model.decoder.roberta.embeddings(generated_tokens)
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
                generated_tokens = torch.cat((generated_tokens, logits[:, -1:].argmax(-1)), dim=1)
                token = logits[:, -1:].argmax(-1)
                if token == tokenizer.eos_token_id:
                    break

            tokens = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True, )
            data[file] = tokens
        json.dump(data, open(f'out/tt/{p}.json', 'w'))


