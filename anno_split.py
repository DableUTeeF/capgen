import pandas as pd
import json


if __name__ == '__main__':
    coco_04 = json.load(open('/mnt/d/work/capgen/capgen_v0.4/annotations/capgen_v0.4_coco.json'))
    train_04 = json.load(open('/mnt/d/work/capgen/capgen_v0.4/annotations/capgen_v0.4_train.json'))
    test_04 = json.load(open('/mnt/d/work/capgen/capgen_v0.4/annotations/capgen_v0.4_test.json'))
    val_04 = json.load(open('/mnt/d/work/capgen/capgen_v0.4/annotations/capgen_v0.4_val.json'))

    solution = {
        'image_id': [],
        'captions': [],
        'Usage': [],
    }
    sams = []
    coco_041 = {}
    for k, v in coco_04.items():
        if k.startswith('test'):
            solution['image_id'].append(k)
            solution['captions'].append(json.dumps(v))
            sams.append(v[0])
            usage = 'Public' if int(k[-1]) % 2 == 0 else 'Private'
            solution['Usage'].append(usage)
        else:
            coco_041[k] = v
    json.dump(coco_041, open('/mnt/d/work/capgen/capgen_v0.4/annotations/capgen_v0.4.1_coco.json', 'w'))
    for k, v in test_04.items():
        k = k.replace('.png', '').replace('.jpg', '')
        solution['image_id'].append(k)
        solution['captions'].append(json.dumps(v))
        usage = 'Public' if int(k[-1]) % 2 == 0 else 'Private'
        solution['Usage'].append(usage)
    solution = pd.DataFrame(solution)
    # solution.to_csv('kaggle/solution.csv', index=False)
    sample = solution[['image_id', 'captions']]
    sample.loc[3:, 'captions'] = ''
    sample.loc[:2, 'captions'] = sams[:3]  # this make no fucking sense but ok I guess
    sample.to_csv('kaggle/sample_submission.csv', index=False)

    # train_041 = {}
    # val_041 = {}
    # for k, v in train_04.items():
    #     k = k.replace('.png', '').replace('.jpg', '')
    #     train_041[k] = v
    # for k, v in val_04.items():
    #     k = k.replace('.png', '').replace('.jpg', '')
    #     val_041[k] = v
    # json.dump(train_041, open('/mnt/d/work/capgen/capgen_v0.4/annotations/capgen_v0.4.1_train.json', 'w'))
    # json.dump(val_041, open('/mnt/d/work/capgen/capgen_v0.4/annotations/capgen_v0.4.1_val.json', 'w'))

