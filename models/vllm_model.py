from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

class LMM():
    def __init__(self):
        model_name = "OpenGVLab/InternVL2-4B"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.llm = LLM(
            model=model_name,
            trust_remote_code=True,
            tensor_parallel_size=2,
            gpu_memory_utilization=0.9
        )

    def query(self, image, text_prompt, temperature=0, top_p=0.95, sample_num=1, max_new_tokens=1024):
        stop_tokens = ["<|endoftext|>", "<|im_start|>", "<|im_end|>", "<|end|>"]
        stop_token_ids = [self.tokenizer.convert_tokens_to_ids(i) for i in stop_tokens]
        self.sampling_params = SamplingParams(
            n=sample_num,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_new_tokens,
            stop_token_ids=stop_token_ids,
            stop='\n```\n'
        )
        messages = [{'role': 'user', 'content': f"<image>\n{text_prompt}"}]

        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)     
        outputs = self.llm.generate({
            "prompt": prompt,
            "multi_modal_data": {"image": image},
        }, self.sampling_params, use_tqdm=False)
        return [output.text for output in outputs[0].outputs]
