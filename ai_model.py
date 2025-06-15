import openai
import os
from dotenv import load_dotenv # used to store secret stuff like API keys

openai.api_key = os.getenv("OPENAI_API_KEY")

models = openai.models.list()
for model in models.data:
    print(model.id)