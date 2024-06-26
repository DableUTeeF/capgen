from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    ViTImageProcessor,
    VisionEncoderDecoderModel,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
)
from PIL import Image, ImageFile
import json
from matplotlib import pyplot as plt
import os
import torch
from tqdm import tqdm


ImageFile.LOAD_TRUNCATED_IMAGES = True


def preprocess(image, image_processor):
    model_inputs = image_processor(images=image, return_tensors='pt')

    return model_inputs


def forward(model, model_inputs, generate_kwargs=None):
    if generate_kwargs is None:
        generate_kwargs = {'max_new_tokens': 200}
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
    model_outputs = forward(model, model_inputs)
    outputs = postprocess(model_outputs)
    return outputs


if __name__ == '__main__':
    image_processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224-in21k')
    models = [
        # 'phayathaibert_v0.3_base_2',
        # 'tinygpt_v0.3_base_2',
        # 'wangchanberta_v0.3_base_2',
        # 'gpt2_v0.3_base_2',
        # 'gpt2_scratch_1e-05_2',
        # 'mgpt_scratch_1e-05_2',
        'mgpt_scratch_2',
    ]
    # test_map = json.load(open('v0.3-v0.4_test_map.json'))
    img_src = '/mnt/d/work/capgen/capgen_v0.4'

    for name in models:
        model = VisionEncoderDecoderModel.from_pretrained(f'/mnt/d/work/capgen/workdir/{name}').eval().cuda()
        tokenizer = AutoTokenizer.from_pretrained(f'/mnt/d/work/capgen/workdir/{name}')
        results = {}
        # for t in ['food', 'travel']:
        #     for file in tqdm(os.listdir(os.path.join(img_src, 'test', t))):
        #         try:
        #             image = Image.open(os.path.join(img_src, 'test', t, file)).convert('RGB')
        #         except:
        #             print('error', file, flush=True)
        #         pred = generate(image)[0]['generated_text']
        #         results[os.path.join('test', file)] = pred
        # json.dump(
        #     results,
        #     open(f'results-kaggle/{name}.json', 'w')
        # )
        for file in tqdm(os.listdir(os.path.join(

                '/mnt/d/work/coco/images/test2017'

        ))):
            try:
                image = Image.open(os.path.join('/mnt/d/work/coco/images/test2017', file)).convert('RGB')
            except:
                print('error', file, flush=True)
            pred = generate(image)[0]['generated_text']
            results[os.path.join('test2017', file)] = pred
        json.dump(
            results,
            open(f'results-kaggle/{name}-coco.json', 'w')
        )
