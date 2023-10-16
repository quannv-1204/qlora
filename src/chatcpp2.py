import time
# from langchain.embeddings import HuggingFaceEmbeddings
from embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma, FAISS
from uuid import uuid4
import gradio as gr
from utils import jload
from ctransformers import AutoModelForCausalLM
from transformers import AutoTokenizer
from sentence_transformers.util import cos_sim
import torch
import numpy as np
from prompt_const import *
import re
from llama_cpp_cuda import Llama

# model = AutoModelForCausalLM.from_pretrained("pretrain/", model_file="fashiongpt-70b-v1.1.Q4_K_M.gguf", gpu_layers=75, context_length=8192)
model = Llama(model_path="pretrain/airoboros-c34b-2.2.1.Q4_K_M.gguf", n_ctx=8192, n_gqa=8, n_gpu_layers=25, rms_norm_eps=1e-5)
tokenizer = AutoTokenizer.from_pretrained("VietnamAIHub/Vietnamese_LLama2_13B_8K_SFT_General_Domain_Knowledge", use_fast=False)

train_datapath = "data/company/final/train_data/training_data.json"
source_datapath = "data/company/final/json/splitted_companydata.json"


embeddings_model_name  = "vinai/vinai-translate-vi2en"
embeddings = HuggingFaceEmbeddings(model_name=embeddings_model_name, device="cuda")



persist_directory_context = "/mnt/sdd/nguyen.van.quan/Researchs/Qlora/data/company/final/db/faiss_context_db"
db_context = FAISS.load_local(persist_directory_context, embeddings)


persist_directory_question = "/mnt/sdd/nguyen.van.quan/Researchs/Qlora/data/company/final/db/input_db"
db_question = Chroma(persist_directory=persist_directory_question, embedding_function=embeddings, collection_metadata={"hnsw:space": "cosine"})


print(f"Successfully loaded the models into memory")


start_message = \
"""### System:
Bạn là trợ lý AI được công ty Sun* phát triển với nhiệm vụ giúp User trả lời các câu hỏi về thông tin nội bộ của công ty Sun*. Hãy trả lời chính xác và chi tiết nhất có thể với thông tin được cung cấp ở context. Chỉ được trả lời với các thông tin trong danh sách context được liệt kê dưới đây. Nếu thông tin trong context không đủ để trả lời, hãy nói tôi không có đủ dữ kiện để trả lời câu hỏi. Không được tạo ra câu trả lời không sử dụng thông tin đã được cho. Nếu có câu hỏi cần đặt ra để trả lời chính xác hơn, hãy đặt câu hỏi. Trả lời user với ngôn ngữ mà họ dùng. Ví dụ khi user hỏi bằng english hãy trả lời với english, khi user hỏi bằng tiếng việt hãy trả lời bằng tiếng việt.

BEGINCONTEXT\n{context}ENDCONTEXT

Current conversation:\n"""

def convert_history_to_text(history, list_context:list):
    context = "".join(
        [
            "".join(
                [
                    f"Document {i+1}:<{item}>\n"
                ]      
            )   for i, item in enumerate(list_context)
        ]
    )
    context = {"context":context}


    if len(history) >= 2:
        sentence = history[-2][1]
        idx = sentence.find("\n\n**")
        sentence = sentence[:idx]
        history[-2][1] = sentence

    text = start_message.format(**context) +"".join(
        [
            "".join(
                [
                    f"### User: {item[0]}\n",
                    f"### Assistant: {item[1]}\n",
                ]
            )
            for item in history[:-1]
        ]
    ) + "\n"
    text += "".join(
        [
            "".join(
                [
                    f"### User:\n{history[-1][0]}\n\n",
                    f"### Assistant:\n",
                ]
            )
        ]
    )
    return text




def _find_keywords(question, max_key_len=4, embeddings=None):

    new_data = []
    tokens = embeddings.tokenc(question, add_special_tokens=False)
    for i in range(len(tokens)-max_key_len+1):
        new_data.append(embeddings.tokdec(tokens[i:i+max_key_len]))
        
    emb_pieces = embeddings.embed_documents(new_data)
    emb_quest = embeddings.embed_query(question)
    similarities = cos_sim(emb_quest, emb_pieces)
    k = 3 if similarities.shape[1] > 2 else 2
    topk = torch.topk(similarities, k)[1][0].tolist()
    new_data = np.asarray(new_data)
    keywords = new_data[topk].tolist()
    return keywords

def find_keywords(question, llm):
    input = {"input": question}
    prompt = KEYWORD_SEARCH_TEMPLATE.format(**input)
    sentence = llm(prompt, max_tokens=128, temperature=0.5, echo=False)
    sentence = sentence['choices'][0]['text']

    pattern = r'"(.*?)"'
    # Use re.findall to extract all matches of the pattern from the sentence
    keywords = re.findall(pattern, sentence)
    keywords.append(question)
    return keywords

def ref_retriever(input, datapath, key=""):
    data = jload(datapath)
    for entry in data:
        if input == entry[key]:
            return entry

def document_ranking_prompt(question, list_contexts):

    data = np.asarray(jload("data/company/processed/contents.json"))
    pieces = jload("data/company/final/json/data_piece.json")
    ids = []
    for i in list_contexts:
        for entry in pieces:
            if i == entry["context"]:
                ids.append(entry["id"])

    ids = list(set(ids))
    retrieved = data[ids]
    retrieved_context = [item['context'].replace('\n', ', ') for item in retrieved]
    docs = "".join(
            [
                "".join(
                    [
                        f"Document {idx}:\n<{item}>\n\n",
                    ]
                )
                for idx, item in enumerate(retrieved_context)
            ]
        ) + f"Question:\n<{question}>\n\n"
    return docs, retrieved

def doc_extracter(sentence, data):
    # Split the sentence into lines
    lines = sentence.split('\n')

    # Initialize an empty list to store JSON objects
    contexts = []
    refs = []

    # Define a regular expression pattern to extract Doc and Relevance values
    pattern = r'Doc: (\d+), Relevance: (\d+)'

    # Iterate through the lines and extract the information
    for line in lines:
        match = re.search(pattern, line)
        if match:
            doc_index = int(match.group(1))
            if doc_index <= len(data):
                context = data[doc_index]['context'].replace('\n', ', ')
                contexts.append(context)
                refs.append(
                    {
                        "title": data[doc_index]['title'],
                        "url": data[doc_index]['url']
                    }
                )

    # Remove duplicate references

    # Convert list of dictionaries to a list of tuples (ignoring the order of elements)
    tuple_list = [tuple(sorted(d.items())) for d in refs]

    # Use a set to remove duplicates and convert back to a list
    unique_tuple_list = list(set(tuple_list))

    # Convert the unique list of tuples back to a list of dictionaries
    unique_refs = [dict(t) for t in unique_tuple_list]


    return contexts, unique_refs


def user(message, history):
    # Append the user's message to the conversation history
    return "", history + [[message, ""]]

def process_example(args):
    for [x, y] in bot(args):
        pass
    return [x, y]


example1 = "Công ty Sun-Asterisk có bao nhiêu loại tài sản ?"
example2 = "Phòng Đào tạo & Phát triển Nhân lực - Learning & Development Line (L&D) có nhiệm vụ gì ?"



def bot(history, temperature, max_new_tokens, top_p, repetition_penalty, context_score):
    cache_input = [doc[0] for doc in db_question.similarity_search_with_relevance_scores(history[-1][0], k=1) if doc[1] >= 0.85]
    if len(cache_input) != 0:
        results = ref_retriever(cache_input[0].page_content, train_datapath, key="input")
        ref = f"\n\n**Reference**:\n - {results['source']}" if results['source'] != "" else ""
        answer = results["output"] + ref
        new_text = ""
        for char in answer:
            new_text += char
            history[-1][1] = new_text
            yield history
    else:
        keywords = find_keywords(history[-1][0], model)
        
        context = []
        for key in keywords:
            inf = db_context.similarity_search_with_relevance_scores(key, k=4, score_threshold=context_score)
            if len(inf) != 0:
                context.extend([i[0].page_content for i in inf])
        
        docs, data_retrieved = document_ranking_prompt(history[-1][0], context)
        docs = {"docs":docs}
        ranking_prompt = RANKING_PROMPT_TEMPLATE.format(**docs)
        ranking_result = model(ranking_prompt, max_tokens=3072, stream=False, echo=False)
        
        contexts, refs = doc_extracter(ranking_result['choices'][0]['text'], data_retrieved)

        ref_text = "\n\n**Reference**: " + "".join(
                    [
                        "".join(
                            [
                                f"\n - [" + entry["title"] + "](" + entry["url"] + ")"
                            ]      
                        )   for entry in refs
                    ]
                ) if len(refs) > 0 else ""

        messages = convert_history_to_text(history, contexts) 
        print(messages)


            

        start = time.time()
        # Initialize an empty string to store the generated text
        partial_text = ""
        for generation_output in model(messages, max_tokens=max_new_tokens, stream=True, temperature=temperature):
            partial_text += generation_output['choices'][0]['text']
            history[-1][1] = partial_text
            yield history
        history[-1][1] += ref_text
        yield history
        end = time.time()
        gentime = end - start
        encoded_prompt = tokenizer.encode(partial_text, return_tensors='pt')
        speed = encoded_prompt.shape[1] / gentime
        print(f"speed: {speed} tokens/s")

def get_uuid():
    return str(uuid4())


with gr.Blocks(
    theme=gr.themes.Soft(),
    css=".disclaimer {font-variant-caps: all-small-caps;}",
) as demo:
    gr.Markdown(
        """<h1><center>Sun* Assistant Demo</center></h1>
"""
    )
    output = gr.Markdown()
    chatbot = gr.Chatbot().style(height=500)
    with gr.Row():
        with gr.Column():
            msg = gr.Textbox(
                label="Chat Message Box",
                placeholder="Chat Message Box",
                show_label=False,
            ).style(container=False)
        with gr.Column():
            with gr.Row():
                submit = gr.Button("Submit")
                stop = gr.Button("Stop")
                clear = gr.Button("Clear")
    with gr.Row():
        with gr.Accordion("Advanced Options:", open=False):
            with gr.Row():
                with gr.Column():
                    with gr.Row():
                        temperature = gr.Slider(
                            label="Temperature",
                            value=0.3,
                            minimum=0.0,
                            maximum=1.0,
                            step=0.1,
                            interactive=True,
                            info="Higher values produce more diverse outputs",
                        )
                with gr.Column():
                    with gr.Row():
                        top_p = gr.Slider(
                            label="Top-p (nucleus sampling)",
                            value=0.9,
                            minimum=0.0,
                            maximum=1,
                            step=0.01,
                            interactive=True,
                            info=(
                                "Sample from the smallest possible set of tokens whose cumulative probability "
                                "exceeds top_p. Set to 1 to disable and sample from all tokens."
                            ),
                        )
                with gr.Column():
                    with gr.Row():
                        max_new_tokens = gr.Slider(
                            label="Max new tokens",
                            value=1024,
                            minimum=0,
                            maximum=4096,
                            step=4,
                            interactive=True,
                            info="The maximum numbers of new tokens",
                        )
                with gr.Column():
                    with gr.Row():
                        repetition_penalty = gr.Slider(
                            label="Repetition Penalty",
                            value=1.15,
                            minimum=1.0,
                            maximum=2.0,
                            step=0.1,
                            interactive=True,
                            info="Penalize repetition — 1.0 to disable.",
                        )
                with gr.Column():
                    with gr.Row():
                        context_score = gr.Slider(
                            label="Context Score",
                            value=0.3,
                            minimum=0.0,
                            maximum=1.0,
                            step=0.01,
                            interactive=True,
                            info="Threshold to retrieve context",
                        )
    with gr.Row():
        gr.Examples(
            examples=[example1, example2],
            inputs=[msg],
            cache_examples=False,
            fn=process_example,
            outputs=[output],
        )
    with gr.Row():
        gr.Markdown(
            "Disclaimer: The model can produce factually incorrect output, and should not be relied on to produce "
            "factually accurate information. The model was trained on various public datasets; while great efforts "
            "have been taken to clean the pretraining data, it is possible that this model could generate lewd, "
            "biased, or otherwise offensive outputs.",
            elem_classes=["disclaimer"],
        )
    with gr.Row():
        gr.Markdown(
            "[Privacy policy](https://gist.github.com/samhavens/c29c68cdcd420a9aa0202d0839876dac)",
            elem_classes=["disclaimer"],
        )

    submit_event = msg.submit(
        fn=user,
        inputs=[msg, chatbot],
        outputs=[msg, chatbot],
        queue=False,
    ).then(
        fn=bot,
        inputs=[
            chatbot,
            temperature,
            max_new_tokens,
            top_p,
            repetition_penalty,
            context_score,
        ],
        outputs=chatbot,
        queue=True,
    )
    submit_click_event = submit.click(
        fn=user,
        inputs=[msg, chatbot],
        outputs=[msg, chatbot],
        queue=False,
    ).then(
        fn=bot,
        inputs=[
            chatbot,
            temperature,
            max_new_tokens,
            top_p,
            repetition_penalty,
            context_score,
        ],
        outputs=chatbot,
        queue=True,
    )
    stop.click(
        fn=None,
        inputs=None,
        outputs=None,
        cancels=[submit_event, submit_click_event],
        queue=False,
    )
    clear.click(lambda: None, None, chatbot, queue=False)

demo.queue(max_size=128, concurrency_count=2)

# Launch your Guanaco Demo!
demo.launch(share=True)
