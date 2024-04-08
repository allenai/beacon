import ast
import together
import time
import os
from ipdb import set_trace

together.api_key = os.getenv("TOGETHER_API_KEY")
model = "togethercomputer/llama-2-70b-chat"
def generate_text(user_input, temperature, max_tokens):
    try: 
        output = together.Complete.create(
        prompt = "<human>:" + user_input + "<bot>:", 
        model = model, 
        max_tokens = 256,
        temperature = temperature,
        top_k = 60,
        top_p = 0.6,
        repetition_penalty = 1,
        stop = ['\n\n']
        )
        response = output['output']['choices'][0]['text']

        return response
     
    except Exception as e:
        retry_time = e.retry_after if hasattr(e, "retry_after") else 30
        print(f"Type Error. Retrying in{retry_time} seconds...")
        time.sleep(retry_time)
        return generate_text(user_input, temperature, max_tokens)


def jsonNotFormattedCorrectly(response):
    try:
        extraction = ast.literal_eval(response.strip())
    except SyntaxError as e:
        # This code will be executed if a SyntaxError exception is raised
        print("This is a Syntax error")
        print(response)
        return True
    except ValueError as e:
        print("This is a Syntax error")
        print(response)
        return True