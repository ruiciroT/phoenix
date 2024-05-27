import os
import time
from getpass import getpass
from typing import Optional

import litellm
import openai
from instrumentor import LitellmInstrumentor, PhoenixLiteLLMHandler

import phoenix as px


def test_litellm() -> None:
    session = px.launch_app()
    if not (openai_api_key := os.getenv("OPENAI_API_KEY")):
        openai_api_key = getpass("🔑 Enter your OpenAI API key: ")
    openai.api_key = openai_api_key
    os.environ["OPENAI_API_KEY"] = openai_api_key
    instrumentor = LitellmInstrumentor()
    callback_handler = PhoenixLiteLLMHandler(instrumentor)
    litellm.success_callback = [callback_handler]
    messages = [{"content": "How can Arize AI help me evaluate my llm models?", "role": "user"}]
    response = litellm.completion(model="gpt-4-turbo-preview", messages=messages)
    print(response)
    # Sleep added to allow for spans to get traced into phoenix while testing locally
    time.sleep(5)
    tds: Optional[px.TraceDataset] = session.get_trace_dataset()
    if tds:
        print(tds.dataframe)


if __name__ == "__main__":
    test_litellm()
