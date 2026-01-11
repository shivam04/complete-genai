import boto3
import json

prompt_data="""
Act as a Shakespeare and write a poem on Generative AI
"""

bedrock=boto3.client(service_name="bedrock-runtime")

payload = {
    "anthropic_version": "bedrock-2023-05-31",
    "max_tokens": 512,
    "temperature": 0.8,
    "top_p": 0.8,
    "messages": [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": prompt_data
                }
            ]
        }
    ]
}
body = json.dumps(payload)
model_id = "anthropic.claude-3-haiku-20240307-v1:0"
response = bedrock.invoke_model(
    modelId=model_id,
    body=body,
    accept="application/json",
    contentType="application/json"
)

response_body = json.loads(response.get("body").read())
response_text = response_body["content"][0]["text"]
print(response_text)