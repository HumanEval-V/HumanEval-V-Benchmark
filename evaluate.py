import ast
import fire
import json
import uuid
import itertools
import numpy as np
from pylint.lint import Run
from pylint.reporters import JSONReporter
from io import StringIO
from collections import defaultdict
from tqdm import tqdm
from datasets import load_dataset
from concurrent.futures import as_completed, ProcessPoolExecutor

from utils import dump_json, load_json, dump_file, delete_file
from execution import check_correctness

dataset = load_dataset('HumanEval-V/HumanEval-V-Benchmark', split='test')

class CodeExtractor(ast.NodeVisitor):
    def __init__(self):
        self.extracted_code = []
        self.parent_stack = []

    def visit_Import(self, node):
        for alias in node.names:
            self.extracted_code.append(f"import {alias.name}" + (f" as {alias.asname}" if alias.asname else ""))
        self.generic_visit(node)

    def visit_ImportFrom(self, node):
        module = node.module if node.module else ""
        for alias in node.names:
            self.extracted_code.append(f"from {module} import {alias.name}" + (f" as {alias.asname}" if alias.asname else ""))
        self.generic_visit(node)

    def visit_FunctionDef(self, node):
        if not self._has_function_or_class_parent():
            self.extracted_code.append(ast.unparse(node))
        self.parent_stack.append(node)
        self.generic_visit(node)
        self.parent_stack.pop()

    def visit_ClassDef(self, node):
        if not self._has_function_or_class_parent():
            self.extracted_code.append(ast.unparse(node))
        self.parent_stack.append(node)
        self.generic_visit(node)
        self.parent_stack.pop()

    def _has_function_or_class_parent(self):
        return any(isinstance(parent, (ast.FunctionDef, ast.ClassDef)) for parent in self.parent_stack)

def extract_code_without_function_class_parent(content):
    tree = ast.parse(content)
    extractor = CodeExtractor()
    extractor.visit(tree)
    return "\n".join(extractor.extracted_code)

def post_process(prediction, signature, test):
    if not signature.endswith('\n'):
        signature += '\n'
    if "```python" in prediction: # ideal situation
        if prediction.count("```python") ==1:
            content = prediction.split("```python")[1].split("```")[0]
        else: # concatenate if there are multiple code blocks
            code_block_splits = prediction.split("```python")
            content = ""
            for code_block_split in code_block_splits:
                if code_block_split.count("```") != 1:
                    continue
                content += code_block_split.split("```")[0]
    elif "``` python" in prediction:
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
        completion = extract_code_without_function_class_parent(content)
    except:
        completion = content
    concatenated_code = signature + "    pass\n\n" + completion + "\n\n" + test
    return completion, concatenated_code

def pylint_check(code):
    file_name = f'cache/{str(uuid.uuid4())}.py'
    dump_file(file_name, code)
    reporter_buffer = StringIO()
    Run([file_name], reporter=JSONReporter(reporter_buffer), exit=False)
    file_results = json.loads(reporter_buffer.getvalue())
    errors = [
        file_result['message'] for file_result in file_results 
            if file_result['type'] in ['error', 'fatal'] 
            and file_result['symbol'] != 'function-redefined'
    ]
    reporter_buffer.close()
    delete_file(file_name)
    return errors

def execution_tasks(task_id, concatenated_code, timeout):
    pylint_errors = pylint_check(concatenated_code)
    if pylint_errors:
        return {
            'task_id': task_id,
            'passed': 0,
            'result': "pylint fail:\n" + '\n'.join(pylint_errors)
        }
    return check_correctness(task_id, concatenated_code, timeout)

def parallel_execution(concatenated_code_samples, timeout=2):
    with ProcessPoolExecutor() as executor:
        futures = []
        results = []
        for task_id, concatenated_code in enumerate(concatenated_code_samples):
            args = (task_id, concatenated_code, timeout)
            future = executor.submit(execution_tasks, *args)
            futures.append(future)

        for future in as_completed(futures):
            result = future.result()
            results.append((result['task_id'], result))
    assert len(results) == len(concatenated_code_samples), "Some problems are not attempted."
    return [i[1] for i in sorted(results, key=lambda x: x[0])]


def retrieve_qid(qid):
    idx = dataset['qid'].index(qid)
    question_data = {
        'qid': qid,
        'function_signature': dataset['function_signature'][idx],
        'test_script': dataset['test_script'][idx],
    }
    return question_data

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
    passed_results_by_qid = defaultdict(list)
    pylint_failed_count = 0
    sample_num = 1
    for item in execution_results:
        qid = item['qid']
        sample_num = len(item['predictions'])
        results = item['results']
        for result in results:
            passed_results_by_qid[qid].append(result['passed'])
            pylint_failed_count += 1 if result['result'].startswith("pylint fail") else 0
            
    k = [1] if sample_num == 1 else [10]
    scores = pass_at_K(passed_results_by_qid, k)
    passed_qids = sorted([k for k, v in passed_results_by_qid.items() if any(v)])
    print(f"Pass@{k[0]}: {scores}")
    print(f"Parsing Success Rate: {round(100*(1 - pylint_failed_count / (sample_num*len(execution_results))), 1)}%")
    print(f"Passed QIDs: {passed_qids}")

def main(prediction_file, score_only=False):
    prediction_data = load_json(prediction_file)
    print(f"\nLoaded {len(prediction_data)} questions for {prediction_file}")
    if not score_only:
        print(f"Executing code solutions with test cases...")
        execution_results = execute_code(prediction_data)
        dump_json(prediction_file.replace(".json", "_executed.json"), execution_results)
    else:
        execution_results = load_json(prediction_file.replace(".json", "_executed.json"))
    compute_score(execution_results)


if __name__ == "__main__":
    fire.Fire(main)
