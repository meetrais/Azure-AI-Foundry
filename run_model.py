import os
from azure.ai.inference import ChatCompletionsClient
from azure.ai.inference.models import SystemMessage, UserMessage
from azure.core.credentials import AzureKeyCredential
from dotenv import load_dotenv

load_dotenv()

endpoint = os.getenv("AZURE_EXISTING_AIPROJECT_ENDPOINT")
api_key = os.getenv("API_KEY")
model_name = "Ministral-3B"

client = ChatCompletionsClient(
    endpoint=endpoint,
    credential=AzureKeyCredential(api_key),
    api_version="2024-05-01-preview"
)

response = client.complete(
    messages=[
        SystemMessage(content="You are a helpful assistant."),
        UserMessage(content="I am going to Paris, what should I see?"),
    ],
    max_tokens=2048,
    temperature=0.8,
    top_p=0.1,
    model=model_name
)

print(response.choices[0].message.content)
