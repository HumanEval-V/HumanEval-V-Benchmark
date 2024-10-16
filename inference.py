import fire
from tqdm import tqdm
from datasets import load_dataset

from models.openai_model import LMM
# from models.vllm_model import LMM
from utils import dump_json

dataset = load_dataset('HumanEval-V/HumanEval-V-Benchmark', split='test')

def main(sample_num, prediction_file):
    results = []
    temperature = 0 if sample_num == 1 else 0.8
    print("Start querying LMM...")
    model = LMM()
    for i in tqdm(range(3)):
        qid = dataset['qid'][i]
        function_signature = dataset['function_signature'][i]
        image = dataset['image'][i]
        predictions = model.query(
            image=image,
            text_prompt=function_signature,
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
