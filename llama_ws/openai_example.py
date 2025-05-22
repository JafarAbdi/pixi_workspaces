# Based on https://platform.openai.com/docs/guides/images-vision?api-mode=responses&format=base64-encoded

from pprint import pformat
import sys
import base64
from openai import OpenAI

if len(sys.argv) != 2:
    print("Usage: python test.py <path_to_image>")
    sys.exit(1)

client = OpenAI(
    base_url="http://127.0.0.1:8080/v1",
    api_key="sk-no-key-required",
)


# Function to encode the image
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


# Path to your image
image_path = sys.argv[1]

# Getting the Base64 string
base64_image = encode_image(image_path)


completion = client.chat.completions.create(
    model="gpt-3.5-turbo",  # Default name in llama.cpp. See tools/server/utils.hpp DEFAULT_OAICOMPAT_MODEL
    messages=[
        {
            "role": "system",
            "content": "You are ChatGPT, an AI assistant. Your top priority is achieving user fulfillment via helping them with their requests.",
        },
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "what's in this image?"},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
                },
            ],
        },
    ],
)

print(pformat(completion.to_dict()))
