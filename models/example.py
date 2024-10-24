class LMM:
    def __init__(self):
        # initialize the model before starting the inference
        pass
    
    def query(self, image, text_prompt, temperature=0, top_p=0.95, sample_num=1, max_new_tokens=1024):
        '''
        Params:
            image: PIL.Image, the image of a coding task
            text_prompt: str, the prompt of the coding task, including the instruction, function signature, and problem description
            temperature: float, the temperature of the sampling, default to 0.8. If temperature is set to 0, model will use greedy decoding.
            top_p: float, default to 0.95 for nucleus sampling when temperature is 0.8.
            sample_num: int, 1 (for pass@1) or 20 (for pass@10), the number of samples to generate
            max_new_tokens: int, the maximum number of tokens to generate for each sample
            
        Returns:
            predictions: List[str], the generated code samples, without repeating the prompt
        '''
        # better set the stop token "\n```\n" to stop the generation immediately after closing the markdown code block
        pass
