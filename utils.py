import os
import json

def make_dir(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


def load_json(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)

def dump_json(file_path, data):
    make_dir(os.path.dirname(file_path))
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=4)


if __name__ == '__main__':
    base_dir = '/cpfs/data/user/zhangfengji/HumanEval4V/HumanEval4V-PipelineV2/output/pixtral_12b_2409'
    questions = os.listdir(base_dir)
    result = []
    for q in questions:
        q_dir = os.path.join(base_dir, q)
        prediction_file = os.path.join(q_dir, 'sample_100.json')
        predictions = load_json(prediction_file)
        predictions['qid'] = predictions['qid'].replace('algo-', '')
        result.append(predictions)
    dump_json('output/pixtral_12b_2409_sample_100.json', result)
    

