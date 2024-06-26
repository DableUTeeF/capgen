import json


if __name__ == '__main__':
    mapping = json.load(open('/mnt/d/work/capgen/capgen_v0.5/mapping.json'))
    coco_04 = json.load(open('/mnt/d/work/capgen/capgen_v0.4/annotations/capgen_v0.4_coco.json'))
    train_04 = json.load(open('/mnt/d/work/capgen/capgen_v0.4/annotations/capgen_v0.4_train.json'))
    test_04 = json.load(open('/mnt/d/work/capgen/capgen_v0.4/annotations/capgen_v0.4_test.json'))
    val_04 = json.load(open('/mnt/d/work/capgen/capgen_v0.4/annotations/capgen_v0.4_val.json'))
    test = {}
    train = {'ipu/'+k: v for k, v in train_04.items()}
    val = {'ipu/'+k: v for k, v in val_04.items()}
    for k, v in mapping.items():
        if 'test2017' in v:
            test[k] = coco_04[v]
        else:
            try:
                test[k] = test_04[v + '.png']
            except:
                test[k] = test_04[v + '.jpg']

    for k, v in coco_04.items():
        if k.startswith('val'):
            val['coco/'+k] = v
        elif k.startswith('train'):
            train['coco/'+k] = v

    json.dump(test, open('/mnt/d/work/capgen/capgen_v0.5/test.json', 'w'))
    json.dump(train, open('/mnt/d/work/capgen/capgen_v0.5/train.json', 'w'))
    json.dump(val, open('/mnt/d/work/capgen/capgen_v0.5/val.json', 'w'))
    print()
