import os
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
from concurrent.futures import as_completed, ProcessPoolExecutor

from utils import dump_json, load_json, dump_file, delete_file, load_data, make_dir
from execution import check_correctness
from inference import EXP_V2C, EXP_V2C_COT, EXP_V2T2C, EXP_V2T2C_4o, EXP_T2C, Experiments


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
        else: # choose the last code block
            code_block_splits = prediction.split("```python")
            content = ""
            possible_content = []
            for code_block_split in code_block_splits[::-1]:
                if code_block_split.count("```") != 1:
                    continue
                if "def solution" in code_block_split:
                    content += code_block_split.split("```")[0]
                    break
                possible_content.append(code_block_split.split("```")[0])
            else:
                content += possible_content[-1] if possible_content else "    pass"
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

def execution_tasks(qid, task_id, concatenated_code, timeout):
    pylint_errors = pylint_check(concatenated_code)
    if pylint_errors:
        return {
            'qid': qid,
            'task_id': task_id,
            'passed': 0,
            'result': "Pylint syntax check failed:\n" + '\n'.join(pylint_errors)
        }
    return check_correctness(qid, task_id, concatenated_code, timeout)

def parallel_execution(concatenated_code_samples, timeout=2):
    execution_results = defaultdict(list)
    executed_count = 0
    with ProcessPoolExecutor() as executor:
        futures = []
        for concatenated_code in concatenated_code_samples:
            qid, task_id, concatenated_code = concatenated_code['qid'], concatenated_code['task_id'], concatenated_code['concatenated_code']
            args = (qid, task_id, concatenated_code, timeout)
            future = executor.submit(execution_tasks, *args)
            futures.append(future)

        for future in tqdm(as_completed(futures), total=len(futures), desc="Executing code solutions"):
            result = future.result()
            executed_count += 1
            execution_results[result['qid']].append((result['task_id'], result))
    assert executed_count == len(concatenated_code_samples), "Some problems are not attempted."
    for qid in execution_results.keys():
        execution_results[qid] = [i[1] for i in sorted(execution_results[qid], key=lambda x: x[0])]
    return execution_results

def retrieve_qid(dataset, qid):
    idx = dataset['qid'].index(qid)
    question_data = {
        'qid': qid,
        'function_signature': dataset['function_signature'][idx],
        'test_script': dataset['test_script'][idx],
    }
    return question_data

def execute_code(dataset, prediction_data):
    final_results_by_qid = dict()
    concatenated_code_samples = []
    for item in prediction_data:
        qid = item['qid']
        question_data = retrieve_qid(dataset, qid)
        predictions = item['predictions']
        item['processed_predictions'] = []
        item['concatenated_predictions'] = []
        for idx, prediction in enumerate(predictions):
            processed_code, concatenated_code = post_process(
                prediction, 
                question_data['function_signature'], 
                question_data['test_script']
            )
            item['processed_predictions'].append(processed_code)
            item['concatenated_predictions'].append(concatenated_code)
            concatenated_code_samples.append({
                'qid': qid,
                'task_id': idx,
                'concatenated_code': concatenated_code
            })
        final_results_by_qid[qid] = item
    
    execution_results = parallel_execution(concatenated_code_samples)
    final_results_list = []
    for qid in final_results_by_qid.keys():
        item = final_results_by_qid[qid]
        item['results'] = execution_results[qid]
        final_results_list.append(item)
    return final_results_list


def pass_at_K(passed_results_by_qid, k):
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
            pylint_failed_count += 1 if result['result'].startswith("Pylint") else 0
            
    k = [1] if sample_num == 1 else [3]
    scores = pass_at_K(passed_results_by_qid, k)
    passed_qids = sorted([k for k, v in passed_results_by_qid.items() if any(v)])
    parse_success_rate = round(100*(1 - pylint_failed_count / (sample_num*len(execution_results))), 1)
    return {
        'scores': scores,
        'parse_success_rate': parse_success_rate,
        # 'passed_qids': passed_qids
    }

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
        if len(predictions) != max_prediction_len:
            print(f"qid: {qid}, len(predictions): {len(predictions)}, max_prediction_len: {max_prediction_len}")
            import ipdb
            ipdb.set_trace()
        aggregated_prediction_data.append({
            'qid': qid,
            'prompts': prompts,
            'predictions': predictions
        })
    return aggregated_prediction_data

def eval_pipeline(prediction_file, score_only=False):
    if not os.path.exists(prediction_file) and not os.path.exists(prediction_file.replace(".json", "_executed.json")):
        raise FileNotFoundError(f"{prediction_file} not found.")
    if not score_only:
        dataset = load_data()
        prediction_data = load_json(prediction_file)
        prediction_data = aggregate_prediction_data(prediction_data)
        assert len(dataset) == len(prediction_data), "Dataset and prediction data have different lengths."
        execution_results = execute_code(dataset, prediction_data)
        dump_json(prediction_file.replace(".json", "_executed.json"), execution_results)
    else:
        execution_results = load_json(prediction_file.replace(".json", "_executed.json"))
    return compute_score(execution_results)

def main(model_name, exp_type, sample_num, exp_base_dir):
    make_dir("cache")
    prediction_file_path = Experiments().get_pred_file_path(model_name, exp_type, sample_num)
    if type(prediction_file_path) == tuple:
        prediction_file_path = prediction_file_path[1]
    prediction_file = os.path.join(exp_base_dir, prediction_file_path)
    if os.path.exists(prediction_file.replace(".json", "_executed.json")):
        print(f"{prediction_file} already executed.")
        result = eval_pipeline(prediction_file, score_only=True)
    else:
        print(f"Executing {prediction_file}")
        result = eval_pipeline(prediction_file)
    print(result)

if __name__ == "__main__":
    fire.Fire(main)
