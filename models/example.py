from utils import load_and_append_json

'''class name must be LMM'''
class LMM:

    def load_model(self):
        """Initialize the model before starting inference."""
        pass

    def query(self, loaded_model, qids, images, text_prompts, prediction_file, sample_num, temperature, top_p=0.95, top_k=20, max_new_tokens=2048):
        """
        Run inference on the given inputs and save the results to the prediction file.

        Args:
            loaded_model: The loaded model to perform inference with.
            qids (List[str]): A list of task IDs (qids) to run inference on.
            images (List[PIL.Image]): A list of images to be processed by the model.
            text_prompts (List[str]): A list of text prompts associated with each task.
            prediction_file (str): The path to the file where the results will be saved.
            sample_num (int): Number of samples to generate, either 1 (for pass@1) or 6 (for pass@3).
            temperature (float, optional): Sampling temperature for the model. A temperature of 0 uses greedy decoding.
            top_p (float, optional): The probability threshold for nucleus sampling (default is 0.95).
            top_k (int, optional): The number of top tokens to sample from (default is 20).
            max_new_tokens (int, optional): The maximum number of tokens to generate for each sample (default is 2048).

        Returns:
            None: The results are saved directly to the prediction file. 
                  You may use the `load_and_append_json` function to append results.
                  The function will create the file if it does not exist.
        """
        pass
