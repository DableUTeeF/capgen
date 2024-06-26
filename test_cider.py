import json
import language_evaluation
from transformers import AutoTokenizer

def tokenize(text):
    return ' '.join([tokenizer.decode(x) for x in tokenizer(text)['input_ids'][1:-1]])


tokenizer = AutoTokenizer.from_pretrained("facebook/nllb-200-distilled-600M", src_lang="tha_Thai")
evaluator = language_evaluation.CocoEvaluator(verbose=False, coco_types=["CIDEr"],)
eng = json.load(open('/mnt/d/work/coco/annotations/captions_val2017.json'))
gt = {}
for ann in eng['annotations']:
    if ann['image_id'] not in gt:
        gt[ann['image_id']] = []
    gt[ann['image_id']].append(tokenize(ann['caption']))

gg = list(gt.values())
g1 = [g[0] for g in gg]
results = evaluator.run_evaluation(g1, gg)
print(results)
