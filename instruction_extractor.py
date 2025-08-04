from strands import Agent
from strands.models import BedrockModel
from models.schemas import InstructionOutput
import boto3

bedrock = BedrockModel(
    model_id="anthropic.claude-sonnet-4-20250514-v1:0",
    boto_session=boto3.Session(region_name="us-west-2"),
    temperature=0.2,
    top_p=0.95
)

class InstructionExtractionAgent(Agent):
    def run(self, inputs):
        pages = inputs["instruction_pages"]
        text = "\n".join([p[1] for p in pages])
        prompt = f"""
Extract structured trade instructions from the following text. Each instruction must include: action, instrument, amount, settlement_date, counterparty.
Respond with JSON:
{{"extracted_instructions": [{{"action": ..., "instrument": ..., "amount": ..., "settlement_date": ..., "counterparty": ...}}]}}

{text}
"""
        return bedrock(prompt, structured=True, schema=InstructionOutput).dict()