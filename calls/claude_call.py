import time
import ast
from ipdb import set_trace
from anthropic import Anthropic, HUMAN_PROMPT, AI_PROMPT

anthropic = Anthropic()

def generate(input, max_tokens_to_sample=512):
    try:
        completion = anthropic.completions.create(
            model="claude-2",
            max_tokens_to_sample=max_tokens_to_sample,
            prompt=f"{HUMAN_PROMPT}{input}{AI_PROMPT}",
        )

    except Exception as e:
        print("Could be any error")
        print(e)
        retry_time = 30
        time.sleep(retry_time)
        return generate(input)

    except anthropic.RateLimitError as e:
        print("A 429 status code was received; we should back off a bit.")
        time.sleep(retry_time)
        return generate(input)

    except anthropic.APIStatusError as e:
        print("API status error was received")
        retry_time = 30
        time.sleep(retry_time)
        return generate(input)

    return completion.completion


def jsonNotFormattedCorrectly(response):
    try:
        extraction = ast.literal_eval(response)
    except SyntaxError as e:
        return True
    except ValueError as e:
        return True
