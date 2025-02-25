import os
import json
from collections import defaultdict

from datasets import load_dataset, load_from_disk

def make_dir(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

def load_json(file_path):
    if not os.path.exists(file_path):
        return []
    with open(file_path, 'r', encoding='utf8') as f:
        return json.load(f)

def load_and_append_json(file_path, data):
    existing_data = load_json(file_path)
    if type(data) != list:
        data = [data]
    existing_data.extend(data)
    dump_json(file_path, existing_data)
    return existing_data

def dump_json(file_path, data):
    make_dir(os.path.dirname(file_path))
    with open(file_path, 'w', encoding='utf8') as f:
        json.dump(data, f, indent=4)

def read_file(file_path):
    with open(file_path, 'r', encoding='utf8') as f:
        return f.read()

def dump_file(file_path, data):
    make_dir(os.path.dirname(file_path))
    with open(file_path, 'w', encoding='utf8') as f:
        f.write(data)

def delete_file(file_path):
    if os.path.exists(file_path):
        os.remove(file_path)

def aggregate_prediction_data(prediction_data):
    # merge the prompts and predictions with the same qid
    prediction_data_by_qid = defaultdict(list)
    for item in prediction_data:
        qid = item['qid']
        prediction_data_by_qid[qid].append(item)
    aggregated_prediction_data = []
    max_prediction_len = 0
    for qid in prediction_data_by_qid.keys():
        prompts = []
        predictions = []
        for item in prediction_data_by_qid[qid]:
            predictions += item['predictions']
            prompts += [item['prompt']] * len(item['predictions'])
        max_prediction_len = max(max_prediction_len, len(predictions))
        assert len(predictions) == max_prediction_len, f"qid: {qid}, len(predictions): {len(predictions)}, max_prediction_len: {max_prediction_len}"
        aggregated_prediction_data.append({
            'qid': qid,
            'prompts': prompts,
            'predictions': predictions
        })
    return aggregated_prediction_data

def load_data(qids_to_exclude=None, generated_diagram_description_path=None, debug=False):
    def load_generated_diagram_description(qids, generated_specification_path):
        generated_specification = load_json(generated_specification_path)
        generated_specification_by_qid = {result['qid']: result['predictions'] for result in generated_specification}
        return [generated_specification_by_qid[qid] for qid in qids]
    
    # humaneval_v = load_dataset("HumanEval-V/HumanEval-V-Benchmark", split="test")
    humaneval_v = load_from_disk("humaneval_v_test_hf")
    
    if debug:
        humaneval_v = humaneval_v.select(range(1))
    
    if qids_to_exclude:
        humaneval_v = humaneval_v.filter(lambda example: example['qid'] not in qids_to_exclude)
    
    if generated_diagram_description_path:
        humaneval_v.generated_diagram_description = load_generated_diagram_description(humaneval_v['qid'], generated_diagram_description_path)
    return humaneval_v
