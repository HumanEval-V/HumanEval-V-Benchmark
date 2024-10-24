import sys
import fire
import importlib

from tqdm import tqdm
from datasets import load_dataset

from utils import dump_json, load_json

def load_model_class(model_name: str):
    module_path = f"models.{model_name}"
    try:
        module = importlib.import_module(module_path)
    except ImportError:
        print(f"Error: Model '{model_name}' not found.")
        sys.exit(1)
    if not hasattr(module, 'LMM'):
        print(f"Error: The module '{module_path}' does not contain a class named 'LMM'.")
        sys.exit(1)
    return getattr(module, 'LMM')

print("Loading dataset...")
dataset = load_dataset('HumanEval-V/HumanEval-V-Benchmark', split='test')

CODE_GEN_TASK_PROMPT = '''**Instructions:**
You are an exceptionally intelligent coding assistant that consistently delivers accurate and reliable responses to user instructions. Please complete the function based on the provided image and code context. Return the complete solution, including the function signature, in a single response, formatted within a Python code block.

**Code Context:**
```python
{signature}
```
'''

def main(model_name, sample_num, prediction_file, temperature=None):
    results = load_json(prediction_file)
    existing_qids = {result['qid'] for result in results}
    if not temperature:
        temperature = 0 if sample_num == 1 else 0.8
    print("Start querying LMM...")
    model = load_model_class(model_name)()
    for i in tqdm(range(len(dataset))):
        qid = dataset['qid'][i]
        if qid in existing_qids:
            print(f"Skip qid {qid}, already queried.")
            continue
        function_signature = dataset['function_signature'][i]
        prompt = CODE_GEN_TASK_PROMPT.format(signature=function_signature)
        image = dataset['image'][i]
        predictions = model.query(
            image=image,
            text_prompt=prompt,
            temperature=temperature,
            sample_num=sample_num
        )
        results.append({
            'qid': qid,
            'predictions': predictions,
        })
        dump_json(prediction_file, results)

if __name__ == '__main__':
    fire.Fire(main)
