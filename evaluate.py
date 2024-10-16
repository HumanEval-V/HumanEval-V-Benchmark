import ast
import fire
import itertools
import numpy as np
from collections import defaultdict
from tqdm import tqdm
from datasets import load_dataset
from concurrent.futures import as_completed, ProcessPoolExecutor

from utils import dump_json, load_json
from execution import check_correctness

dataset = load_dataset('HumanEval-V/HumanEval-V-Benchmark', split='test')

def retrieve_qid(qid):
    idx = dataset['qid'].index(qid)
    question_data = {
        'qid': qid,
        'function_signature': dataset['function_signature'][idx],
        'test_script': dataset['test_script'][idx],
    }
    return question_data

def post_process(prediction, signature, test):
    if not signature.endswith('\n'):
        signature += '\n'
    if "```python" in prediction: # ideal situation
        content = prediction.split("```python")[1].split("```")[0]
    elif "``` python" in prediction: # internvl2 40b
        content = prediction.split("``` python")[1].split("```")[0]
    else:
        count = prediction.count("```")
        if count == 1: # only generated the question body
            content = prediction.split("```")[0]
        elif count >= 2: # surround code in ``` ```, not using python indicator
            content = prediction.split("```")[1]
        else:
            content = "    pass"
    
    if content.startswith("\n  ") or content.startswith("  "): # without function signature
        content = signature + content
    try:
        tree = ast.parse(content)
        extracted_code = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    extracted_code.append(f"import {alias.name}" + (f" as {alias.asname}" if alias.asname else ""))
            elif isinstance(node, ast.ImportFrom):
                module = node.module if node.module else ""
                for alias in node.names:
                    extracted_code.append(f"from {module} import {alias.name}" + (f" as {alias.asname}" if alias.asname else ""))
            elif isinstance(node, ast.FunctionDef):
                extracted_code.append(ast.unparse(node))
        completion = "\n".join(extracted_code)
    except:
        completion = '''
def AST_parse_fail():
    assert False, "Fail to parse as a complete code."
AST_parse_fail()
'''
    concatenated_code = signature + "    pass\n\n" + completion + "\n\n" + test
    return completion, concatenated_code


def parallel_execution(concatenated_code_samples, timeout=2):
    with ProcessPoolExecutor() as executor:
        futures = []
        results = []
        for pred_id, concatenated_code in enumerate(concatenated_code_samples):
            args = (pred_id, concatenated_code, timeout)
            future = executor.submit(check_correctness, *args)
            futures.append(future)
        
        for future in as_completed(futures):
            result = future.result()
            results.append((result['task_id'], result))
    assert len(results) == len(concatenated_code_samples), "Some problems are not attempted."
    return [i[1] for i in sorted(results, key=lambda x: x[0])]

def execute_code(prediction_data):
    execution_results = []
    for item in tqdm(prediction_data):
        qid = item['qid']
        question_data = retrieve_qid(qid)
        predictions = item['predictions']
        processed_code_samples = []
        concatenated_code_samples = []
        for prediction in predictions:
            processed_code, concatenated_code = post_process(
                prediction, 
                question_data['function_signature'], 
                question_data['test_script']
            )
            concatenated_code_samples.append(concatenated_code)
            processed_code_samples.append(processed_code)
        results = parallel_execution(concatenated_code_samples)
        item['results'] = results
        item['processed_predictions'] = processed_code_samples
        item['concatenated_predictions'] = concatenated_code_samples
        execution_results.append(item)
    return execution_results


def pass_at_K(passed_results_by_qid, k=[1, 10]):
    # Calculate pass@k.
    total, correct = [], []
    for passed in passed_results_by_qid.values():
        total.append(len(passed))
        correct.append(sum(passed))

    total = np.array(total)
    correct = np.array(correct)

    ks = k
    return {f"pass@{k}": round(float(_estimate_pass_at_k(total, correct, k).mean())*100, 1)
            for k in ks if (total >= k).all()}

def _estimator(n: int, c: int, k: int) -> float:
    """
    Calculates comb(n - c, k) / comb(n, k).
    """
    if n - c < k:
        return 0
    return np.prod(1.0 - k / np.arange(n - c + 1, n + 1))

def _estimate_pass_at_k(num_samples, num_correct, k) -> np.ndarray:
    """
    Estimates pass@k of each problem and returns them in an array.
    """
    if isinstance(num_samples, int):
        num_samples_it = itertools.repeat(num_samples, len(num_correct))
    else:
        assert len(num_samples) == len(num_correct)
        num_samples_it = iter(num_samples)

    return np.array([1.0 - _estimator(int(n), int(c), k) for n, c in zip(num_samples_it, num_correct)])

def compute_score(execution_results):
    results = []
    passed_results_by_qid = defaultdict(list)
    sample_num = 1
    for item in execution_results:
        qid = item['qid']
        sample_num = len(item['predictions'])
        results = item['results']
        for result in results:
            passed_results_by_qid[qid].append(result['passed'])
            
    k = [1] if sample_num == 1 else [10]
    scores = pass_at_K(passed_results_by_qid, k)
    passed_qids = sorted([k for k, v in passed_results_by_qid.items() if any(v)])
    print(f"Pass@{k[0]}: {scores}")
    print(f"Passed QIDs: {passed_qids}")

def main(prediction_file, score_only=False):
    prediction_data = load_json(prediction_file)
    if not score_only:
        print(f"Executing code solutions with test cases...")
        execution_results = execute_code(prediction_data)
        dump_json(prediction_file.replace(".json", "_executed.json"), execution_results)
    else:
        execution_results = load_json(prediction_file.replace(".json", "_executed.json"))
    compute_score(execution_results)


if __name__ == "__main__":
    fire.Fire(main)
