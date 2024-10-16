from openai import OpenAI
import base64
from io import BytesIO

class LMM:
    def __init__(self):
        self.client = OpenAI(
            api_key="your_api_key_here",
        )

    def make_interleave_content(self, image, text_prompt):
        content = []
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

    def query(self, image, text_prompt, temperature=0, top_p=0.95, sample_num=1, max_new_tokens=1024):
        model = "gpt-4o-mini"
        new_messages = [
            {
                "role": "user",
                "content": self.make_interleave_content(image, text_prompt),
            }  
        ]
        response = self.client.chat.completions.create(
            model=model,
            messages=new_messages,
            max_tokens=max_new_tokens,
            timeout=60,
            top_p=top_p,
            n=sample_num,
            temperature=temperature,
            stop=['\n```\n']
        )
        predictions = [response.choices[i].message.content for i in range(len(response.choices))]
        return predictions