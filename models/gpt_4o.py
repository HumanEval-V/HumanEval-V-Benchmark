from openai import OpenAI
import base64
import time
from io import BytesIO
from concurrent.futures import ProcessPoolExecutor, as_completed

from utils import load_and_append_json

class LMM:
    def load_model(self):
        return 'gpt-4o'
    
    @staticmethod
    def single_query(loaded_model, qid, image, text_prompt, sample_num, temperature, top_p, top_k, max_new_tokens):
        def make_interleave_content(image, text_prompt):
            content = []
            if image:
                byte_io = BytesIO()
                image.save(byte_io, format="PNG")
                byte_data = byte_io.getvalue()
                base64_image = base64.b64encode(byte_data).decode("utf-8")
                image_elem = {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_image}",
                        "detail": "auto",
                    },
                }
                content.append(image_elem)
            text_elem = {
                "type": "text",
                "text": text_prompt,
            }
            content.append(text_elem)
            return content
        
        client= OpenAI(
            api_key="Your API key here",
        )
        new_messages = [
            {
                "role": "user",
                "content": make_interleave_content(image, text_prompt),
            }  
        ]
        try_count = 0
        while try_count < 10:
            try:
                try_count += 1
                response = client.chat.completions.create(
                    model=loaded_model,
                    messages=new_messages,
                    max_tokens=max_new_tokens,
                    timeout=60,
                    top_p=top_p,
                    # top_k=top_k, # top_k is not supported in the API
                    n=sample_num,
                    temperature=temperature
                )
                predictions = [response.choices[i].message.content for i in range(len(response.choices))]
                return {
                    "qid": qid,
                    "prompt": text_prompt,
                    "predictions": predictions,
                }
            except Exception as e:
                print(e)
                time.sleep(60)
        if try_count >= 10:
            exit(1)
    
    def query(self, loaded_model, qids, images, text_prompts, prediction_file, sample_num, temperature, top_p=0.95, top_k=20, max_new_tokens=2048):
        with ProcessPoolExecutor(max_workers=8) as executor:
            start_time = time.time()
            count = 0
            
            futures = []
            for qid, image, text_prompt in zip(qids, images, text_prompts):
                param = (loaded_model, qid, image, text_prompt, sample_num, temperature, top_p, top_k, max_new_tokens, )
                futures.append(executor.submit(LMM.single_query, *param))

            for job in as_completed(futures):
                results_to_save = job.result(timeout=None)
                end_time = time.time()
                average_time = (end_time - start_time) / (count + 1)
                count += 1
                print(f"Processed {count}/{len(qids)} questions, average time per question: {average_time:.2f}s, expected time: {average_time/60 * (len(futures) - count):.2f}min")
                data = load_and_append_json(prediction_file, results_to_save)
                print(f"Saved {len(data)} results to {prediction_file}")