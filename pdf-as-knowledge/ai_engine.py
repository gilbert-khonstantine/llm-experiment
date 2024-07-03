import torch
from langchain import HuggingFacePipeline
from transformers import pipeline, AutoModelForQuestionAnswering
from huggingface_hub import login
from dotenv import dotenv_values

config = dotenv_values(".env")
login(token = config.get("HUGGING_FACE_KEY"))


def qna_with_context(context, question):
 
    model_name = "deepset/roberta-base-squad2"

    # a) Get predictions
    nlp = pipeline('question-answering', model=model_name, tokenizer=model_name)
    QA_input = {
        'question': question,
        'context': context
    }
    res = nlp(QA_input)
    return res
