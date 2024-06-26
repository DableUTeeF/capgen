from transformers import (
    Blip2ForConditionalGeneration,
    AutoConfig,
    BlipProcessor,
)
from PIL import Image, ImageFile
import json
import os
import torch
ImageFile.LOAD_TRUNCATED_IMAGES = True


@torch.no_grad()
def generate(model, processor, name):
    src = '/project/lt200203-aimedi/ipu24-v0.5/test'
    results = {}
    for file in os.listdir(src):
        image = Image.open(os.path.join(src, file)).convert('RGB')
        inputs = processor(image, return_tensors="pt").to("cuda", torch.float16)
        out = model.generate(**inputs, max_new_tokens=100)
        text = processor.decode(out[0], skip_special_tokens=True)
        results[file] = text
    json.dump(results, open(f'results-v0.5/{name}', 'w'))


if __name__ == '__main__':
    models = [
        'blip_frozen_phayathaibert_blip2-opt-2.7b-coco',
    ]
    for name in models:
        blip_model = f'/mnt/d/work/capgen/workdir/{name}'
        text_decode_model = 'clicknext/phayathaibert'
        cfg = AutoConfig.from_pretrained(blip_model)
        lm_cfg = AutoConfig.from_pretrained(text_decode_model)
        lm_cfg.is_decoder = True
        cfg.text_config = lm_cfg
        cfg.use_decoder_only_language_model = True
        blip_ori = Blip2ForConditionalGeneration.from_pretrained(blip_model)
        blip_ori.language_model = None
        blimp = Blip2ForConditionalGeneration(cfg)
        blimp.vision_model.load_state_dict(blip_ori.vision_model.state_dict())
        blimp.qformer.load_state_dict(blip_ori.qformer.state_dict())
        del blip_ori
        generate(
            blimp.to("cuda"),
            BlipProcessor.from_pretrained("/project/lt200203-aimedi/palm/huggingface/blip-image-captioning-large"),
            'blip-large'
        )

        print()
