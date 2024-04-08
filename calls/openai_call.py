import os
import ast
import time
import openai

openai.api_key = os.getenv("OPENAI_API_KEY")

def generate_text(user_input, temperature, max_tokens):
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": user_input}],
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0,
        )
        extracted_output = response["choices"][0]["message"]["content"]
        return extracted_output

    except openai.error.Timeout as e:
        retry_time = e.retry_after if hasattr(e, "retry_after") else 30
        print(f"Rate limit exceeded. Retrying in {retry_time} seconds...")
        time.sleep(retry_time)
        return generate_text(user_input, temperature, max_tokens)

    except openai.error.RateLimitError as e:
        retry_time = e.retry_after if hasattr(e, "retry_after") else 30
        print(f"Rate limit exceeded. Retrying in {retry_time} seconds...")
        time.sleep(retry_time)
        return generate_text(user_input, temperature, max_tokens)

    except openai.error.ServiceUnavailableError as e:
        retry_time = e.retry_after if hasattr(e, "retry_after") else 30
        print(f"Service Unavailable. Retrying in {retry_time} seconds...")
        time.sleep(retry_time)
        return generate_text(user_input, temperature, max_tokens)

    except openai.error.APIError as e:
        retry_time = e.retry_after if hasattr(e, "retry_after") else 30
        print(f"API Error. Retrying in {retry_time} seconds...")
        time.sleep(retry_time)
        return generate_text(user_input, temperature, max_tokens)


def jsonNotFormattedCorrectly(response):
    try:
        extraction = ast.literal_eval(response)
    except SyntaxError as e:
        # This code will be executed if a SyntaxError exception is raised
        print(e)
        print("This is a Syntax error")
        return True
