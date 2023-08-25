# Load the model.
# Note: It can take a while to download LLaMA and add the adapter modules.
# You can also use the 13B model by loading in 4bits.
import sys
sys.path.insert(1, '/mnt/sdd/nguyen.van.quan/Researchs/Qlora/')
from transformers import AutoTokenizer, pipeline, logging, TextIteratorStreamer, AutoModelForCausalLM
from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig
import time
import torch 
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from privateGPT.constants import CHROMA_SETTINGS 
import os
import datetime
from threading import Event, Thread
from uuid import uuid4
import gradio as gr
import requests
from utils.exllama_model import ExllamaModel
import argparse
from glob import glob
from utils.utils import jdump, jload
def get_args():
    parser = argparse.ArgumentParser()                               
    parser.add_argument('--gpu_split', type=str, default="17.2, 24", help="Comma-separated list of VRAM (in GB) to use per GPU device for model layers, e.g. 20,7,7")
    parser.add_argument('--max_seq_len', type=int, default=4096, help="Maximum sequence length.")
    parser.add_argument('--compress_pos_emb', type=int, default=1, help="Positional embeddings compression factor. Should typically be set to max_seq_len / 2048.")
    parser.add_argument('--alpha_value', type=int, default=1, help="Positional embeddings alpha factor for NTK RoPE scaling. Scaling is not identical to embedding compression. Use either this or compress_pos_emb, not both.")
    return parser.parse_args()

args = get_args()

embeddings_model_name = "intfloat/e5-large-v2"
persist_directory = "/mnt/sdd/nguyen.van.quan/Researchs/Qlora/privateGPT/db"
target_source_chunks = 3
model_path = "output/custom-13b/_quantize_model" 

model, tokenizer = ExllamaModel.from_pretrained(model_path, args)

embeddings = HuggingFaceEmbeddings(model_name=embeddings_model_name)
db = Chroma(persist_directory=persist_directory, embedding_function=embeddings, client_settings=CHROMA_SETTINGS)
retriever = db.as_retriever(search_type="similarity_score_threshold",
                            search_kwargs={"k":target_source_chunks, "score_threshold":0.7})


print(f"Successfully loaded the model {model_path} into memory")



def convert_history_to_text(question, context):
    start_message = \
    f"""### System:\nYou are an AI assistant helping a human keep track of facts about relevant people, things, and events happening all over the world.  You are able to process and understand large amounts of text, and can use this knowledge to provide accurate and informative responses to a wide range of questions. Additionally, you are able to generate your own text based on the input you receive, allowing you to engage in discussions and provide explanations and descriptions on a wide range of topics. You can choose yourself whether to use the context provided in the quotation marks below or not, because some pieces of context may not relate to the question.\n\n
    Context:\n"{context}"\n\n"""

    text = start_message + "".join(
        [
            "".join(
                [
                    f"### User:\nPlease use the Context to answer this question briefly: {question}\n\n",
                    f"### Assistant:\n",
                ]
            )
        ]
    )
    return text





list_path = sorted(glob('data/test_json/*'))
article_dir = "data/data_new/"

for i,path in enumerate(list_path):
    data = jload(path)
    name = path.split("/")[-1].split(".")[0] + ".txt"
    for atr in data:
        query = atr["input"]
        context = open(article_dir+name).read()
        prompt = convert_history_to_text(query, context)
        print("\n\n",prompt)
        generate_kwargs = dict(
        temperature=0.7,
        top_p=0.9,
        repetition_penalty=1.15,
        ban_eos_token=False,
        top_k=49,
        typical_p=1.,
        max_new_tokens=512,
    )
        bot_answer = model.generate(prompt, generate_kwargs)
        print("bot answer: ", bot_answer)
        print("gold answer: ", atr['output'])
        atr["bot_answer"] = bot_answer
    jdump(data, path)