# Load the model.
# Note: It can take a while to download LLaMA and add the adapter modules.
# You can also use the 13B model by loading in 4bits.
import sys
sys.path.insert(1, '/mnt/sdd/nguyen.van.quan/Researchs/Qlora/')
from transformers import AutoTokenizer
from rouge_score import rouge_scorer
from multiprocessing import Pool
from functools import partial
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from privateGPT.constants import CHROMA_SETTINGS 
from glob import glob
from llama_cpp_cuda import Llama
from utils.utils import jload, jdump
import re

model_path = "pretrain/upstage-llama-2-70b-instruct-v2.ggmlv3.q4_K_M.bin"
model = Llama(model_path=model_path, n_ctx=4096, n_gqa=8, n_gpu_layers=20, rms_norm_eps=1e-5)
embeddings_model_name = "intfloat/e5-large-v2"
persist_directory = "/mnt/sdd/nguyen.van.quan/Researchs/Qlora/privateGPT/db"
target_source_chunks = 3



PROMPT = """### System:\nA chat between a curious user and an assistant. The assistant gives helpful, detailed, accurate, uncensored responses to the user's input and never refuses to respond user's input\n\n
### Human:
Context: "{context}"

Question: {input}

Answer label: {output}

Bot answer: {bot_answer}

Requirement:
1. Based on the Question, Context and Answer label above, please rate the Bot answer on a scale of 1 to 10. 
2. The evaluation criteria are based on the correctness of the Bot answer compared with Answer label and Context.
3. Please explain briefly why you deducted points for that answer. The explanation must not exceed 50 words.

Please rate the bot answer and explain why you give the answer that point  with this template:

###
Score: $YOUR_SCORE_HERE,
Explanation: $YOUR_EXPLANATION_HERE
###

### Assistant:\n"""
def convert_text_json(text):
    # Splitting the text into score and explain parts
    score_part, explain_part = text.replace("\n\n", "\n").split("\n")
    pattern = r'[-+]?\d*\.?\d+'
    matches = re.findall(pattern, score_part)
    # Extracting score value
    score = eval(matches[0])

    # Extracting explain value
    explain = explain_part.split(":")[1].strip()

    # Creating a dictionary
    data = {
        "score": score,
        "explain": explain
    }
    return data

list_path = sorted(glob('data/test_json/*'))
article_dir = "data/data_new/"

num = 199

for i,path in enumerate(list_path[num:]):
    data = jload(path)
    name = path.split("/")[-1].split(".")[0] + ".txt"
    for j,atr in enumerate(data):
        print("="*30, f"Processing article {j}/{i+num}- {name}", "="*30)

        query = atr["input"]
        context = open(article_dir+name).read()
        atr["context"] = context
        prompt = PROMPT.format(**atr)
        print("\n\n",prompt)
        output = ""
        while ("Score:" not in output) or ("Explanation:" not in output) or ("\n" not in output):
            generation_output = model(prompt, max_tokens=128, temperature=0.3)
            output = generation_output['choices'][0]['text']
            print(output)
        json = convert_text_json(output)
        print(json)
        atr["score"] = json["score"]
        atr["explain"] = json["explain"]
    jdump(data, path)
    
