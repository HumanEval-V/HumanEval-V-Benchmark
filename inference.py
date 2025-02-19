import os
import sys
import fire
import importlib
from functools import partial

from utils import dump_json, load_json, load_data, delete_file, make_dir
from prompts import PROMPT_V2C, PROMPT_V2C_WITH_COT, PROMPT_T2C, PROMPT_V2T, PROMPT_ITER_V2C, PROMPT_ITER_V2T

# experiment types
EXP_V2C = 'V2C'
EXP_V2C_COT = 'V2C-CoT'
EXP_V2T2C = 'V2T2C'
EXP_V2T2C_4o = 'V2T2C-4o'
EXP_T2C = 'GT-T2C'

class QueryRunner:
    def run_v2c_prompting(self, dataset, query_func, cot):
        signatures = dataset['function_signature']
        diagrams = dataset['diagram']
        prompt_template = PROMPT_V2C_WITH_COT if cot else PROMPT_V2C
        prompts = [prompt_template.format(function_signature=signature) for signature in signatures]
        return query_func(qids=dataset['qid'], images=diagrams, text_prompts=prompts)

    def run_t2c_prompting(self, dataset, query_func):
        signatures = dataset['function_signature']
        if hasattr(dataset, 'generated_diagram_description'):
            if len(dataset.generated_diagram_description) == 0:
                return
            diagram_descriptions = dataset.generated_diagram_description # 2d list
            specification_sample_num = len(diagram_descriptions[0])
            for spec in diagram_descriptions:
                assert len(spec) == specification_sample_num, f"Each qid should have {specification_sample_num} generated diagram description."

            qids = [qid for qid in dataset['qid'] for _ in range(specification_sample_num)]
            diagrams = [None]*len(qids)
            prompts = [
                PROMPT_T2C.format(function_signature=signature, problem_specification=diagram_description) 
                for signature, diagram_description_inner in zip(signatures, diagram_descriptions)
                    for diagram_description in diagram_description_inner
            ]
            return query_func(qids=qids, images=diagrams, text_prompts=prompts)
        else:
            diagram_descriptions = dataset['ground_truth_diagram_description']
            prompts = [
                PROMPT_T2C.format(function_signature=signature, problem_specification=diagram_description)
                for signature, diagram_description in zip(signatures, diagram_descriptions)
            ]
            diagrams = [None]*len(prompts)
            return query_func(qids=dataset['qid'], images=diagrams, text_prompts=prompts)

    def run_v2t_prompting(self, dataset, query_func):
        signatures = dataset['function_signature']
        prompts = [PROMPT_V2T.format(function_signature=signature) for signature in signatures]
        diagrams = dataset['diagram']
        return query_func(qids=dataset['qid'], images=diagrams, text_prompts=prompts)

class Experiments:
    def get_pred_file_path(self, model_name, exp_type, sample_num):
        template = "{model_name}_{exp_type}_sample_{sample_num}.json"
        if exp_type in [EXP_V2C, EXP_V2C_COT, EXP_T2C]:
            return template.format(model_name=model_name, exp_type=exp_type, sample_num=sample_num)
        elif exp_type in [EXP_V2T2C, EXP_V2T2C_4o]:
            return (
                f'{model_name}_V2T_sample_{sample_num}.json',
                f'{model_name}_{exp_type}_T2C_sample_{sample_num}.json'
            )
            
    def load_model_class(self, model_name):
        module_path = f"models.{model_name}"
        try:
            module = importlib.import_module(module_path)
        except ImportError as e:
            print(f"Error: {e}")
            sys.exit(1)
        if not hasattr(module, 'LMM'):
            print(f"Error: The module '{module_path}' does not contain a class named 'LMM'.")
            sys.exit(1)
        return getattr(module, 'LMM')

    def get_model_query_func(self, model, loaded_model, sample_num, prediction_file):
        params = {
            'temperature': 0 if sample_num == 1 else 0.8,
            'sample_num': sample_num,
            'top_p': 0.95,
            'top_k': 20,
            'max_new_tokens': 2048
        }
        return partial(model.query, loaded_model=loaded_model, prediction_file=prediction_file,  **params)

    def load_data_to_run(self, prediction_file, generated_diagram_description_path, overwrite):
        if overwrite:
            delete_file(prediction_file)
        previous_prediction = load_json(prediction_file)
        existing_qids = {result['qid'] for result in previous_prediction}
        dataset = load_data(qids_to_exclude=existing_qids, generated_diagram_description_path=generated_diagram_description_path)
        print(f"{len(existing_qids)} tasks already been queried. {len(dataset['qid'])} tasks to run,")
        if not dataset['qid']:
            print("No tasks to query.")
        return dataset

    def run_v2c_experiments(self, model_name, sample_num, exp_base_dir, cot, overwrite=False):
        exp_type = EXP_V2C if not cot else EXP_V2C_COT
        prediction_file = self.get_pred_file_path(model_name, exp_type, sample_num)
        assert type(prediction_file) == str, "Invalid prediction file path."
        prediction_file = os.path.join(exp_base_dir, prediction_file)
        
        model = self.load_model_class(model_name)()
        loaded_model = model.load_model()
        
        generated_diagram_description_path = None
        dataset = self.load_data_to_run(prediction_file, generated_diagram_description_path, overwrite)
        query_func = self.get_model_query_func(model, loaded_model, sample_num, prediction_file)
        runner = QueryRunner().run_v2c_prompting
        runner(dataset, query_func, cot)
        
    def run_v2t2c_experiments(self, model_name, sample_num, exp_base_dir, with_strong_coder, overwrite=False):
        exp_type = EXP_V2T2C if not with_strong_coder else EXP_V2T2C_4o
        v2t_prediction_file, t2c_prediction_file = self.get_pred_file_path(model_name, exp_type, sample_num)
        v2t_prediction_file = os.path.join(exp_base_dir, v2t_prediction_file)
        t2c_prediction_file = os.path.join(exp_base_dir, t2c_prediction_file)

        model = self.load_model_class(model_name)()
        loaded_model = model.load_model()

        # first to generate diagram descriptions
        generated_diagram_description_path = None
        dataset = self.load_data_to_run(v2t_prediction_file, generated_diagram_description_path, overwrite)
        query_func = self.get_model_query_func(model, loaded_model, sample_num, v2t_prediction_file)
        runner = QueryRunner().run_v2t_prompting
        runner(dataset, query_func)
        
        # then to generate code
        if with_strong_coder:
            model = self.load_model_class('gpt_4o')()
            loaded_model = model.load_model()

        dataset = self.load_data_to_run(t2c_prediction_file, v2t_prediction_file, overwrite)
        query_func = self.get_model_query_func(model, loaded_model, 1, t2c_prediction_file) # sample 1 for code generation
        runner = QueryRunner().run_t2c_prompting
        runner(dataset, query_func)

    def run_gt_t2c_experiments(self, model_name, sample_num, exp_base_dir, overwrite=False):
        prediction_file = self.get_pred_file_path(model_name, EXP_T2C, sample_num)
        assert type(prediction_file) == str, "Invalid prediction file path."
        prediction_file = os.path.join(exp_base_dir, prediction_file)
        model = self.load_model_class(model_name)()
        loaded_model = model.load_model()
        generated_diagram_description_path = None
        dataset = self.load_data_to_run(prediction_file, generated_diagram_description_path, overwrite)
        query_func = self.get_model_query_func(model, loaded_model, sample_num, prediction_file)
        runner = QueryRunner().run_t2c_prompting
        runner(dataset, query_func)


def main(exp_type, model_name, exp_base_dir, sample_num, overwrite=False):
    make_dir(exp_base_dir)
    if exp_type not in [EXP_V2C, EXP_V2C_COT, EXP_T2C, EXP_V2T2C, EXP_V2T2C_4o]:
        print(f"Error: Invalid experiment type '{exp_type}'.")
        sys.exit(1)
    exp = Experiments()
    if exp_type in [EXP_V2C, EXP_V2C_COT]:
        exp.run_v2c_experiments(model_name, sample_num, exp_base_dir, exp_type==EXP_V2C_COT, overwrite)
    elif exp_type in [EXP_V2T2C, EXP_V2T2C_4o]:
        exp.run_v2t2c_experiments(model_name, sample_num, exp_base_dir, exp_type==EXP_V2T2C_4o, overwrite)
    elif exp_type == EXP_T2C:
        exp.run_gt_t2c_experiments(model_name, sample_num, exp_base_dir, overwrite)

if __name__ == '__main__':
    fire.Fire(main)
