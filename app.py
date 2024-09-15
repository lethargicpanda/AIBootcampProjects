import chainlit as cl
import openai
import os
from prompts import SYSTEM_PROMPT
from langsmith.wrappers import wrap_openai
from langsmith import traceable
from dotenv import load_dotenv
from langsmith import Client
from langchain_openai import ChatOpenAI

load_dotenv()

#api_key = os.getenv("OPEN_API_KEY")
api_key = os.getenv("MISTRAL_API_KEY")

# endpoint_url = "https://api.openai.com/v1"
endpoint_url = "https://api.mistral.ai/v1/"

langsmith_api_key = os.getenv("LANGSMITH_API_KEY")


# Client(api_key=langsmith_api_key)
client = wrap_openai(openai.AsyncClient(api_key=api_key, base_url=endpoint_url))
# client = wrap_openai(OpenAI())

# https://platform.openai.com/docs/models/gpt-4o
# model_kwargs = {
#     "model": "chatgpt-4o-latest",
#     "temperature": 0.3,
#     "max_tokens": 500
# }

model_kwargs = {
    "model": "open-mixtral-8x7b",
    # "temperature": 0.3,
    # "max_tokens": 500
}

@cl.on_message
@traceable
async def on_message(message: cl.Message):
    message_history = cl.user_session.get("message_history", [])
    
    if (not message_history or message_history[0].get("role") != "system"):
        system_prompt_content = SYSTEM_PROMPT
        message_history.insert(0, {"role": "system", "content": system_prompt_content})
    
    message_history.append({"role": "user", "content": message.content})

    response_message = cl.Message(content="")
    await response_message.send()

    stream = await client.chat.completions.create(
        messages=message_history, 
        stream=True, **model_kwargs
    )

    async for part in stream:
        if token := part.choices[0].delta.content or "":
            await response_message.stream_token(token)

    await response_message.update()

    # Record the AI;s resposne in the history
    message_history.append({"role": "assistant", "content": response_message.content})
    cl.user_session.set("message_history", message_history)