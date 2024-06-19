import os
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, GPTNeoXForCausalLM, GPTNeoXTokenizerFast, \
    GPTNeoForCausalLM, GPT2Tokenizer, AutoModelForCausalLM
import openai

import utils

openai.api_key = os.getenv("OPENAI_API_KEY")


class OpenAIGPT:
    model = "gpt-3.5-turbo-instruct"
    #model = "text-davinci-003"
    #model = "gpt-4"
    seconds_per_query = (60 / 20) + 0.01
    @staticmethod
    def request_model(prompt):
        return openai.Completion.create(model=OpenAIGPT.model, prompt=prompt, max_tokens=400, temperature=0.0, top_p=1, frequency_penalty=0, presence_penalty=0, best_of=1)

    @staticmethod
    def request_chat_model(msgs):
        messages = []
        for message in msgs:
            content, role = message
            messages.append({"role": role, "content": content})
        return openai.ChatCompletion.create(model=OpenAIGPT.model, messages=messages)

    @staticmethod
    def decode_response(response):
        if OpenAIGPT.is_chat():
            return response["choices"][0]["message"]["content"]
        else:
            return response["choices"][0]["text"]

    @staticmethod
    def query(prompt):
        return OpenAIGPT.decode_response(OpenAIGPT.request_model(prompt))

    @staticmethod
    def chat_query(msgs):
        return OpenAIGPT.decode_response(OpenAIGPT.request_chat_model(msgs))

    @staticmethod
    def is_chat():
        return OpenAIGPT.model in ["gpt-4"]

    @staticmethod
    def __call__(inputs):
        if OpenAIGPT.is_chat():
            return OpenAIGPT.chat_query(inputs)
        else:
            return OpenAIGPT.query(inputs)
