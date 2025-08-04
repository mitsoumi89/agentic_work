from pydantic import BaseModel, Field
from typing import List

class EmailPlannerOutput(BaseModel):
    relevant_attachments: List[str]
    analyze_email_body: bool = True

class ClassificationOutput(BaseModel):
    category: str

class PageExtractionOutput(BaseModel):
    instruction_pages: List[int]

class Instruction(BaseModel):
    action: str
    instrument: str
    amount: str
    settlement_date: str
    counterparty: str

class InstructionOutput(BaseModel):
    extracted_instructions: List[Instruction]