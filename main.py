from strands import Strand
from agents.planner import EmailPlannerAgent
from agents.sampler import SamplerAgent
from agents.classifier import ClassifierAgent
from agents.page_extractor import PageExtractionAgent
from agents.instruction_extractor import InstructionExtractionAgent
import concurrent.futures

strand = Strand([
    SamplerAgent(id="sampler"),
    ClassifierAgent(id="classifier"),
    PageExtractionAgent(id="extractor"),
    InstructionExtractionAgent(id="parser")
])

def process_attachment(name, file_bytes):
    context = {"document": file_bytes}
    out1 = strand.get_agent("sampler").run(context)
    out2 = strand.get_agent("classifier").run(out1)
    if out2["category"] != "Trade":
        return []
    out3 = strand.get_agent("extractor").run(context)
    out4 = strand.get_agent("parser").run(out3)
    return out4["extracted_instructions"]

def process_email(email_body: str, attachments: dict):
    planner = EmailPlannerAgent(id="planner")
    plan = planner.run({"email_body": email_body, "attachments": attachments})

    all_results = []
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [executor.submit(process_attachment, name, attachments[name])
                   for name in plan["relevant_attachments"]]
        for future in concurrent.futures.as_completed(futures):
            all_results.extend(future.result())

    return {"instructions": all_results}

# Example usage
if __name__ == "__main__":
    attachments = {
        "file1.pdf": open("example_trade_doc.pdf", "rb").read(),
        "file2.pdf": open("static_change_doc.pdf", "rb").read()
    }
    email_text = "Dear team, please find attached the subscription and redemption confirmations."
    output = process_email(email_text, attachments)
    print(output)