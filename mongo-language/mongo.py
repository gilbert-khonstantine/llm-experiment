import torch
from langchain import HuggingFacePipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig, pipeline, AutoModelForQuestionAnswering, BitsAndBytesConfig
from huggingface_hub import login
login(token = "TOKEN")

print("hello world")
def llama2_model(prompt):
    MODEL_NAME = "meta-llama/Llama-2-7b-hf"
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
    
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.float16, trust_remote_code=True, device_map="auto"
    )
    
    generation_config = GenerationConfig.from_pretrained(MODEL_NAME)
    generation_config.max_new_tokens = 1024
    generation_config.temperature = 0.0001
    generation_config.top_p = 0.95
    generation_config.do_sample = True
    generation_config.repetition_penalty = 1.15
    
    text_pipeline = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        generation_config=generation_config,
    )
    
    llm = HuggingFacePipeline(pipeline=text_pipeline, model_kwargs={"temperature": 0})

    return llm(prompt)

###
# print(llama2_model('What is Machine Learning?'))

def qna_distilled_bert_model(prompt, context):
    model_name = "distilbert/distilbert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForQuestionAnswering.from_pretrained(model_name)
    question_answerer = pipeline("question-answering", model=model, tokenizer= tokenizer)
    return question_answerer(prompt, context)

###
# question = "How many programming languages does BLOOM support?"
# context = "BLOOM has 176 billion parameters and can generate text in 46 languages natural languages and 13 programming languages."
# print(qna_distilled_bert_model(question, context))

def causal_distilled_bert_model(prompt):
    model_name = "distilbert/distilgpt2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    generation_config = GenerationConfig.from_pretrained(model_name)
    generation_config.max_new_tokens = 1024
    generation_config.temperature = 0.0001
    generation_config.top_p = 0.95
    generation_config.do_sample = True
    generation_config.repetition_penalty = 1.15

    generator = pipeline("text-generation", model=model, tokenizer= tokenizer, generation_config = generation_config)

    return generator(prompt)

###
# print(causal_distilled_bert_model("""You are a database expert in mongo query language. Your task is to generate a Mongo query of a user question starts and ends with single quote. The Mongo schema is defined as follows {executedBy, tradeDate, quantity, notional, createdBy} in ORDERS document
#                                   'How many trades are created by John?'
#                                   """))


def code_llama_model(prompt):
 
    model_id =  "codellama/CodeLlama-7b-hf"
    
    model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto")
    
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    eos_token_id = tokenizer.eos_token_id
    
    eval_prompt = """### Task
    # Generate a MongoDB query to answer the following question:
    # `{question}`
    
    # ### Database Schema
    # This query will run on a Mongo Database with ORDERS document name, whose schema is represented in this string:
    
    # '''executedBy: {{"person who executes the trade", tradeDate: "when trade is happening", quantity: "quantity of the trades", notional: "nominal amount of the trade with currency", createdBy: "who created this record}}"'''
    
    # An example of the Mongo Query would be 'db.vehicle.find({{"createdBy": "James"}})'
    
    # ### Mongo Query
    # Given the database schema, here is the Mongo query that answers `{question}`:
    # ```mongo
    # """.format(question=prompt)
    
    model_input = tokenizer(eval_prompt, return_tensors="pt")
    
    #model.eval()
    outputs = tokenizer.batch_decode(model.generate(**model_input,eos_token_id=eos_token_id,pad_token_id=eos_token_id,max_new_tokens=100,do_sample=False,
        num_beams=1), skip_special_tokens=True)
    print(outputs[0].split("```mongo")[-1], reindent=True)
    return generator(prompt)

###
# print(code_llama_model("Can you tell me about the orders which are completed two days ago?"))



def bloom_model(prompt):
 
    model_id =  "bigscience/bloom-560m"
    
    model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto")
    
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    eos_token_id = tokenizer.eos_token_id
    
    eval_prompt = """### Task
    # Generate a MongoDB query to answer the following question:
    # `{question}`
    
    # ### Database Schema
    # This query will run on a Mongo Database with ORDERS document name, whose schema is represented in this string:
    
    # '''executedBy: {{"person who executes the trade", tradeDate: "when trade is happening", quantity: "quantity of the trades", notional: "nominal amount of the trade with currency", createdBy: "who created this record}}"'''
    
    # An example of the Mongo Query would be 'db.vehicle.find({{"createdBy": "James"}})'
    
    # ### Mongo Query
    # Given the database schema, here is the Mongo query that answers `{question}`:
    # ```mongo
    # """.format(question=prompt)
    
    model_input = tokenizer(eval_prompt, return_tensors="pt")
    
    #model.eval()
    outputs = tokenizer.batch_decode(model.generate(**model_input,eos_token_id=eos_token_id,pad_token_id=eos_token_id,max_new_tokens=100,do_sample=False,
        num_beams=1), skip_special_tokens=True)
    return outputs[0].split("```mongo")[-1]

###
print(bloom_model("Can you tell me about the orders which are completed two days ago?"))


