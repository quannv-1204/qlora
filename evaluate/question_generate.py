# Load the model.
# Note: It can take a while to download LLaMA and add the adapter modules.
# You can also use the 13B model by loading in 4bits.

from transformers import AutoTokenizer
from rouge_score import rouge_scorer
from multiprocessing import Pool
from functools import partial

from exllama_model import ExllamaModel
import argparse
from llama_cpp_cuda import Llama

model_path = "pretrain/airoboros-ggml/stablebeluga2-70b.ggmlv3.q4_K_M.bin"
model = Llama(model_path=model_path, n_ctx=4096, n_gqa=8, n_gpu_layers=60, rms_norm_eps=1e-5)
print(f"Successfully loaded the model {model_path} into memory")


def encode_prompt(prompt_instructions):
    """Encode multiple prompt instructions into a single string."""
    prompt = '"' + open(prompt_instructions).read() + '"' + "\n\n" + \
        open("utils/prompt_question.txt").read() + "\n"
    return prompt

def convert_history_to_text(message, human="Human", assistant="Assistant"):
    start_message = """A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions.\n\n"""
    
    text = start_message + "".join(
        [
            "".join(
                [
                    f"### {human}: {message}\n\n",
                    f"### {assistant}:",
                ]
            )

        ]
    )

    return text

from glob import glob 
import os 

num_questions = 5
num_cpus=8
output_dir = "data/test_questions/"
list_dir = sorted(glob('data/data_700tok/*'))
for i, article in enumerate(list_dir[43:]): 

    article_name = article.split('/')[-1].split('.')[0]
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    # if os.path.exists(f'{output_dir}{article_name}.txt'):
    #     continue
    
    message = encode_prompt(article)
    
    prompt = convert_history_to_text(message)
    print(f"=======Processing article {i}: {article_name}=======")
    print(prompt)
    # similarities = {}
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)

    num = 0
    qa = ""
    previous_quests_token = []

    while num < num_questions:
        generation_output = model(prompt, max_tokens=1024)
        generation_output = generation_output['choices'][0]['text'].replace("\n\n", "\n")
        list_qa = generation_output.split("\n")
        # previous_quests_token.extend([scorer._tokenizer.tokenize(q) for q in list_qa])
        # print(previous_quests_token)
        # for quest in list_qa:
        #     new_quest_token = scorer._tokenizer.tokenize(quest)
            
        #     with Pool(num_cpus) as p:
        #         rouge_scores = p.map(partial(rouge_scorer._score_lcs, new_quest_token),
        #                             previous_quests_token)
        #     rouge_scores = [score.fmeasure for score in rouge_scores]
        #     if max(rouge_scores) > 0.7 or quest.isspace():
        #         continue
        #     else:
        #         num += 1
            # qa += quest + "\n"
            # print(quest)
            # previous_quests_token.append(new_quest_token)
        num += len(list_qa)
        qa += generation_output + "\n"
    print(qa)
    with open(f'{output_dir}{article_name}.txt', 'w') as fp:
        fp.write(qa)
    
