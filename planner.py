from strands import Agent
from tools.pdf_utils import sample_pdf_pages
from strands.models import BedrockModel
from models.schemas import EmailPlannerOutput
import boto3

bedrock = BedrockModel(
    model_id="anthropic.claude-sonnet-4-20250514-v1:0",
    boto_session=boto3.Session(region_name="us-west-2"),
    temperature=0.2,
    top_p=0.95
)

def chunk_dict(d, size):
    it = iter(d.items())
    for _ in range(0, len(d), size):
        yield dict([next(it) for _ in range(min(size, len(d)))])

class EmailPlannerAgent(Agent):
    def run(self, inputs, chunk_size=10):
        email_body = inputs["email_body"]
        attachments = inputs["attachments"]
        all_relevant = set()

        for chunk in chunk_dict(attachments, chunk_size):
            attachment_samples = {
                name: sample_pdf_pages(file_bytes)
                for name, file_bytes in chunk.items()
            }

            context = {
                "email_body": email_body,
                "attachments": [
                    {"name": name, "sample": sample}
                    for name, sample in attachment_samples.items()
                ]
            }

            prompt = f"""
You are an assistant helping to plan the analysis of an email containing trade documents.
Given the email body and a list of attachment samples, identify which attachments likely contain trade instructions (subscription, redemption, switch).
Respond with JSON:
{{"relevant_attachments": ["filename1.pdf", "filename2.pdf"]}}

{context}
"""
            result = bedrock(prompt, structured=True, schema=EmailPlannerOutput)
            all_relevant.update(result.relevant_attachments)

        return {"relevant_attachments": list(all_relevant)}