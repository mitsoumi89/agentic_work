from strands import Agent
from tools.pdf_utils import extract_all_pages
from strands.models import BedrockModel
from models.schemas import PageExtractionOutput
import boto3

bedrock = BedrockModel(
    model_id="anthropic.claude-sonnet-4-20250514-v1:0",
    boto_session=boto3.Session(region_name="us-west-2"),
    temperature=0.2,
    top_p=0.95
)

class PageExtractionAgent(Agent):
    def run(self, inputs):
        all_pages = extract_all_pages(inputs["document"])
        pages_input = "\n".join([f"Page {i}: {txt}" for i, txt in all_pages])
        prompt = f"""
Select the pages from the document that contain trade instructions (subscription/redemption/switch).
Respond with JSON: {{"instruction_pages": [list of page numbers]}}

{pages_input}
"""
        response = bedrock(prompt, structured=True, schema=PageExtractionOutput).dict()
        instruction_pages = [p for p in all_pages if p[0] in response["instruction_pages"]]
        return {"instruction_pages": instruction_pages}