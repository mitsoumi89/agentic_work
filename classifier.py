from strands import Agent
from strands.models import BedrockModel
from models.schemas import ClassificationOutput
import boto3

bedrock = BedrockModel(
    model_id="anthropic.claude-sonnet-4-20250514-v1:0",
    boto_session=boto3.Session(region_name="us-west-2"),
    temperature=0.2,
    top_p=0.95
)

class ClassifierAgent(Agent):
    def run(self, inputs):
        prompt = f"""
Classify this document into one of: Trade, Static Data Change, Query, Other.
Respond with JSON: {{"category": "<category>"}}

{inputs['sample_text']}
"""
        return bedrock(prompt, structured=True, schema=ClassificationOutput).dict()