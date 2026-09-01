import asyncio
import os
import time

from openai import AsyncOpenAI


MODEL_NAME = os.environ.get("OPENAI_MODEL", "gpt-4.1-mini")


def get_openai_client():
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set.")
    return AsyncOpenAI(api_key=api_key)


async def dispatch_openai_requests(model_name, messages_list, temperature):
    client = get_openai_client()
    tasks = [
        client.chat.completions.create(
            model=model_name,
            messages=messages,
            temperature=temperature,
        )
        for messages in messages_list
    ]
    return await asyncio.gather(*tasks)


def call_async(samples, wrap_gen_message, print_result=False):
    message_list = []
    for sample in samples:
        message_list.append(wrap_gen_message(sample))

    try:
        predictions = asyncio.run(
            dispatch_openai_requests(
                model_name=MODEL_NAME,
                messages_list=message_list,
                temperature=0.0,
            )
        )
    except Exception as exc:
        print(f"Error in call_async: {exc}")
        time.sleep(6)
        return []

    results = []
    for sample, prediction in zip(samples, predictions):
        if prediction:
            content = prediction.choices[0].message.content
            sample["result"] = content
            if print_result:
                print(content)
            results.append(sample)
    return results
