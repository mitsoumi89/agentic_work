from strands import Agent
from tools.pdf_utils import sample_pdf_pages

class SamplerAgent(Agent):
    def run(self, inputs):
        file_bytes = inputs["document"]
        sample_text = sample_pdf_pages(file_bytes)
        return {"sample_text": sample_text}