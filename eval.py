from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langsmith.evaluation import evaluate, LangChainStringEvaluator
from langsmith.schemas import Run, Example
import json
import os
import openai

from dotenv import load_dotenv
load_dotenv()

from langsmith.wrappers import wrap_openai
from langsmith import traceable

api_key = os.getenv("MISTRAL_API_KEY")
endpoint_url = "https://api.mistral.ai/v1/"
client = wrap_openai(openai.AsyncClient(api_key=api_key, base_url=endpoint_url))

@traceable
def prompt_compliance_evaluator(run: Run, example: Example) -> dict:
    inputs = example.inputs['input']
    outputs = example.outputs['output']

    # Extract system prompt
    system_prompt = next((msg['data']['content'] for msg in inputs if msg['type'] == 'system'), "")

    # Extract message history
    message_history = []
    for msg in inputs:
        if msg['type'] in ['human', 'ai']:
            message_history.append({
                "role": "user" if msg['type'] == 'human' else "assistant",
                "content": msg['data']['content']
            })

    # Extract latest user message and model output
    latest_message = message_history[-1]['content'] if message_history else ""
    model_output = outputs['data']['content']


    evaluation_prompt = f"""
    System Prompt: {system_prompt}

    Message History:
    {json.dumps(message_history, indent=2)}

    Latest User Message: {latest_message}

    Model Output: {model_output}

    Based on the above information, evaluate the model's output for compliance with the system prompt and context of the conversation. 
    Provide a score from 0 to 5 using the following scale:
    0: the model output is completely non-compliant 
    1: the model output is mostly non-compliant
    2: the model output is somewhat compliant but misses most key aspects of the question 
    3: the model output is somewhat compliant but misses a few key aspects of the question 
    4: the model output is compliant but doesn't completely answer the question
    5: the model output is perfectly compliant.

    Provide a brief explanation for your score.

    Respond in the following JSON format:
    {{
        "explanation": "<string>",
        "score": <int>
    }}
    """

    response = client.chat.completions.create(
        #model="gpt-4o",
        model="open-mixtral-8x7b",
        messages=[
            {"role": "system", "content": "You are an AI assistant tasked with evaluating the compliance of model outputs to given prompts and conversation context."},
            {"role": "user", "content": evaluation_prompt}
        ],
        temperature=0.2
    )

    try:
        result = json.loads(response.choices[0].message.content)
        return {
            "key": "prompt_compliance",
            "score": result["score"] / 5,  # Normalize to 0-1 range
            "reason": result["explanation"]
        }
    except json.JSONDecodeError:
        return {
            "key": "prompt_compliance",
            "score": 0,
            "reason": "Failed to parse evaluator response"
        }

# The name or UUID of the LangSmith dataset to evaluate on.
data = "summarizer_0"

# A string to prefix the experiment name with.
experiment_prefix = "Summarizer prompt compliance"

# List of evaluators to score the outputs of target task
evaluators = [
    prompt_compliance_evaluator
]

# Evaluate the target task
results = evaluate(
    lambda inputs: inputs,
    data=data,
    evaluators=evaluators,
    experiment_prefix=experiment_prefix,
)

print(results)