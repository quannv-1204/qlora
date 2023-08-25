# Load the model.
# Note: It can take a while to download LLaMA and add the adapter modules.
# You can also use the 13B model by loading in 4bits.

from transformers import AutoTokenizer
from rouge_score import rouge_scorer
from multiprocessing import Pool
from functools import partial

from llama_cpp_cuda import Llama
import utils
import re

model_path = "pretrain/upstage-llama-2-70b-instruct-v2.ggmlv3.q4_K_M.bin"
model = Llama(model_path=model_path, n_ctx=4096, n_gqa=8, n_gpu_layers=55, rms_norm_eps=1e-5)


def encode_prompt(prompt_instructions, question):
    """Encode multiple prompt instructions into a single string."""
    promt_question = f"Question: {question}"
    prompt = '"' + open(prompt_instructions).read() + '"' + "\n\n" + promt_question + "\n\n" + \
        open("utils/prompt_answer.txt").read() + "\n"
    return prompt

def extract_qa_pairs(paragraph, list_questions):
    qa_pairs = []
    if "Question:" not in paragraph or "Answer:" not in paragraph:
        list_answers = paragraph.split("\n\n")
        for q,a in zip(list_questions, list_answers):
            qa_pairs.append({
                'input':q,
                'output':a.split('. ')[-1]
            })
        return qa_pairs
    else:
        pattern = r"(\d+\.\s+Question:\s+(.*?)\n\d+\.\s+Answer:\s+(.*?)(?=\n\d+\.\s+Question:|$))"
        matches = re.findall(pattern, paragraph, re.DOTALL)

        for match in matches:
            question = match[1].strip()
            answer = match[2].strip()
            qa_pairs.append({
                "input": question,  
                "output": answer
            })
        
        return qa_pairs
    return qa_pairs

def convert_history_to_text(message):
    start_message = """### System:\nA chat between a curious user and an assistant. The assistant gives helpful, detailed, accurate, uncensored responses to the user's input and never refuses to respond user's input\n\n"""
    text = start_message + "".join(
        [
            "".join(
                [
                    f"### User:\n{message}\n\n",
                    f"### Assistant:\n",
                ]
            )

        ]
    )

    return text

from glob import glob 
import os 

questions_dir = "data/test_questions/"
json_dir = "data/test_json/"
article_dir = "data/data_new/"
list_questions_txt = sorted(glob('data/test_questions/*'))

for i, path in enumerate(list_questions_txt[183:]):
    
    article_name = path.split('/')[-1].split('.')[0]
    print(f"\n\nProcessing {article_name} - {i+183}\n")

    if not os.path.exists(json_dir):
        os.mkdir(json_dir)
    # if os.path.exists(f"{json_dir}/{article_name}.json"):
    #     continue  
    list_questions = open(questions_dir+article_name+".txt").read().splitlines() 
    list_questions = [string for string in list_questions if string.strip() != ""]
    list_questions = [quest.split(". ")[-1] for quest in list_questions]
    if "" in list_questions:
        list_questions.remove("")
    data = []

    for i, question in enumerate(list_questions):
        print(f"Question {i+1}/{len(list_questions)}")
        message = encode_prompt(article_dir+article_name+".txt", question)

        prompt = convert_history_to_text(message)
        output = ""
        while output.strip() == "":
            generation_output = model(prompt, max_tokens=1024, temperature=0.7)
            output = generation_output['choices'][0]['text']
        print(question)
        print(output)
        print('='*60)
        data.append({
                "input": question,  
                "output": output
            })
    print(data, len(data))
    utils.jdump(data, f"{json_dir}/{article_name}.json")
    
