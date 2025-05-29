from transformers import AutoTokenizer, AutoModelForQuestionAnswering
import torch

MODEL_PATH = "distilbert-base-uncased-distilled-squad"

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForQuestionAnswering.from_pretrained(MODEL_PATH)

def get_answer(question, context):
    inputs = tokenizer(question, context, return_tensors="pt", truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
        start_idx = torch.argmax(outputs.start_logits)
        end_idx = torch.argmax(outputs.end_logits) + 1
        answer_tokens = inputs["input_ids"][0][start_idx:end_idx]
        return tokenizer.decode(answer_tokens, skip_special_tokens=True)
