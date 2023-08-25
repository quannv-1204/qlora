#!/usr/bin/env python3
from dotenv import load_dotenv
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.vectorstores import Chroma
import os
import argparse
import time
from langchain.vectorstores import DeepLake
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory, ChatMessageHistory
from custom_QLoRA import LangchainQLoRA
from constants import CHROMA_SETTINGS

load_dotenv()

embeddings_model_name = os.environ.get("EMBEDDINGS_MODEL_NAME")
persist_directory = os.environ.get('PERSIST_DIRECTORY')

model_path = os.environ.get("MODEL_PATH")
model_basename = os.environ.get("MODEL_BASENAME")
adapter_path = os.environ.get("ADAPTER_PATH")
n_ctx = os.environ.get("N_CTX")
max_new_tokens = os.environ.get("MAX_NEW_TOKENS")
temp = os.environ.get("TEMPERATURE")
repetition_penalty = os.environ.get("REPETITION_PENALTY")
top_p = os.environ.get("TOP_P")
target_source_chunks = int(os.environ.get('TARGET_SOURCE_CHUNKS'))

prompt_template = \
"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.
    
### Instruction:\nYou are an AI assistant helping a human keep track of facts about relevant people, things, and events happening all over the world, please use the following pieces of context and conversation history to answer the question about world news. But be careful, some pieces of context may be redundant and not relate to the question. If you do not know the answer to a question, just truthfully say you do not know.

Context:
{context}

Current conversation:
{history}

### Input: \n{question}
    
### Response: \n"""



def main():
    # Parse the command line arguments
    args = parse_arguments()
    embeddings = HuggingFaceEmbeddings(model_name=embeddings_model_name)
    db = Chroma(persist_directory=persist_directory, embedding_function=embeddings, client_settings=CHROMA_SETTINGS)
    retriever = db.as_retriever(search_type="similarity_score_threshold",
                                search_kwargs={"k":target_source_chunks, "score_threshold":0.3})
    
    question = "What caused the evacuation of residents in Dunedin, Florida?"
    print(retriever.get_relevant_documents(question)[0].page_content)
    # retriever = db.as_retriever(search_kwargs={"k": target_source_chunks})
    # activate/deactivate the streaming StdOut callback for LLMs
    callbacks = [] if args.mute_stream else [StreamingStdOutCallbackHandler()]
    # Prepare the LLM
    llm = LangchainQLoRA(
        model_path=model_path,
        model_basename=model_basename,
        adapter_path=adapter_path,
        n_ctx=n_ctx,
        max_new_tokens=max_new_tokens,
        temperature=temp,
        repetition_penalty=repetition_penalty,
        top_p=top_p,
        callbacks=callbacks
    )

    PROMPT = PromptTemplate(
        template=prompt_template, input_variables=["history", "context", "question"]
    )
    qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever,
                                     return_source_documents=not args.hide_source,
                                     verbose=True,
                                     chain_type_kwargs={
                                        "verbose": True,
                                        "prompt": PROMPT,
                                        "memory": ConversationBufferMemory(
                                                    memory_key="history",
                                                    input_key="question",
                                                    human_prefix='### Input', 
                                                    ai_prefix='### Response'),
                                     })
    # Interactive questions and answers
    while True:
        query = input("\nEnter a query: ")
        if query == "exit":
            break
        if query.strip() == "":
            continue

        # Get the answer from the chain
        start = time.time()
        res = qa(query)
        answer, docs = res['result'], [] if args.hide_source else res['source_documents']
        end = time.time()

        # Print the result
        print("\n\n> Question:")
        print(query)
        print(f"\n> Answer (took {round(end - start, 2)} s.):")
        print(answer)

        # Print the relevant sources used for the answer
        # for document in docs:
        #     print("\n> " + document.metadata["source"] + ":")
        #     print(document.page_content)
        

def parse_arguments():
    parser = argparse.ArgumentParser(
        description='privateGPT: Ask questions to your documents without an internet connection, '
                    'using the power of LLMs.')
    parser.add_argument("--hide-source", "-S", action='store_true',
                        help='Use this flag to disable printing of source documents used for answers.')

    parser.add_argument("--mute-stream", "-M",
                        action='store_true',
                        help='Use this flag to disable the streaming StdOut callback for LLMs.')

    return parser.parse_args()


if __name__ == "__main__":
    main()
