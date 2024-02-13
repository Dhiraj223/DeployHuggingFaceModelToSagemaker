import json
import logging
import torch
from typing import List
from sagemaker_inference import encoder
from transformers import AutoTokenizer, AutoModelForQuestionAnswering

ANSWER = "answer"

class QuestionAnswer:

    def __init__(self) -> None:
        model_name = "timpal0l/mdeberta-v3-base-squad2"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForQuestionAnswering.from_pretrained(model_name)
        logging.info(f"Commander I am Here")


    def __call__(self, question: str,context: str) -> str :
        
        with torch.inference_mode():
            inputs = self.tokenizer(question,context,return_tensors="pt")
            inputs.pop("labels", None)
            outputs = self.model(**inputs)
            start_logits = outputs.start_logits
            end_logits = outputs.end_logits

            # Find the answer span
            start_index = torch.argmax(start_logits)
            end_index = torch.argmax(end_logits)
            answer = self.tokenizer.convert_tokens_to_string(self.tokenizer.convert_ids_to_tokens(inputs["input_ids"][0][start_index:end_index+1]))

        return answer
    
def model_fn(model_dir: str) -> QuestionAnswer:
    try:
        return QuestionAnswer()
    except Exception :
        logging.exception(f"Failed to load model from: {model_dir}")
        raise

def transform_fn(question_answer: QuestionAnswer, input_data: bytes, content_type: str, accept: str) -> bytes:
    payload = json.loads(input_data)
    question = payload["question"]
    context = payload["context"]
    model_output = question_answer(question=question,context=context)
    output = {ANSWER : model_output}
    return encoder.encode(output,accept)