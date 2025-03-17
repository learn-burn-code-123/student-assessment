#!/usr/bin/env python3
import os
from dotenv import load_dotenv

# Load environment variables
if os.path.exists('.env.test'):
    load_dotenv('.env.test')
    print("Loaded environment variables from .env.test")
else:
    load_dotenv()
    print("Loaded environment variables from .env")

# Get the token
hf_api_token = os.getenv('HF_API_TOKEN')
print(f"Token: '{hf_api_token}'")
print(f"Token is None: {hf_api_token is None}")
print(f"Token strip is empty: {hf_api_token.strip() == '' if hf_api_token else 'N/A'}")
print(f"Token validation would return: {hf_api_token is not None and hf_api_token.strip() != ''}")
