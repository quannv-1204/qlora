from typing import Any, Dict, List, Optional

import requests

from langchain.embeddings.base import Embeddings
from langchain.pydantic_v1 import BaseModel, Extra, Field
from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F
from torch import Tensor

import json
import logging
import os
import shutil
import stat
from collections import OrderedDict
from typing import List, Dict, Tuple, Iterable, Type, Union, Callable, Optional
import requests
import numpy as np
from numpy import ndarray
import torch
from torch import nn, Tensor, device
import torch.multiprocessing as mp
from tqdm.autonotebook import trange
import math
import queue
import tempfile
from distutils.dir_util import copy_tree

logger = logging.getLogger(__name__)





class CustomViEmbeddings(nn.Module):
    """HuggingFace sentence_transformers embedding models.

    To use, you should have the ``sentence_transformers`` python package installed.

    Example:
        .. code-block:: python

            from utils.embeddings import HuggingFaceEmbeddings


            hf = CustomViEmbeddings(
                model_name=model_name,
                device="cuda",
            )
    """

    
    



    def __init__(self, model_name: str = "vinai/vinai-translate-vi2en",
                 device: str = "cuda"):
        """Initialize the sentence_transformer."""
        super().__init__()

        self.model = AutoModel.from_pretrained(model_name).to(device)
        self.tokenizer = AutoTokenizer.from_pretrained("vinai/vinai-translate-vi2en", src_lang="vi_VN")
        self.device = device

    def average_pool(self, last_hidden_states: Tensor,
                    attention_mask: Tensor) -> Tensor:
        last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
        return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]


    def encode(self, sentences: Union[str, List[str]],
               batch_size: int = 32,
               convert_to_numpy: bool = True,
               convert_to_tensor: bool = False,
               normalize_embeddings: bool = True) -> Union[List[Tensor], ndarray, Tensor]:
        """
        Computes sentence embeddings

        :param sentences: the sentences to embed
        :param batch_size: the batch size used for the computation
        :param convert_to_numpy: If true, the output is a list of numpy vectors. Else, it is a list of pytorch tensors.
        :param convert_to_tensor: If true, you get one large tensor as return. Overwrites any setting from convert_to_numpy
        :param device: Which torch.device to use for the computation
        :param normalize_embeddings: If set to true, returned vectors will have length 1. In that case, the faster dot-product (util.dot_score) instead of cosine similarity can be used.

        :return:
           By default, a list of tensors is returned. If convert_to_tensor, a stacked tensor is returned. If convert_to_numpy, a numpy matrix is returned.
        """
        input_was_string = False
        if isinstance(sentences, str) or not hasattr(sentences, '__len__'): #Cast an individual sentence to a list with length 1
            sentences = [sentences]
            input_was_string = True

        all_embeddings = []
        length_sorted_idx = np.argsort([-self._text_length(sen) for sen in sentences])
        sentences_sorted = [sentences[idx] for idx in length_sorted_idx]

        for start_index in trange(0, len(sentences), batch_size, desc="Batches", disable=True):
            sentences_batch = sentences_sorted[start_index:start_index+batch_size]
            features = self.tokenizer(sentences_batch, padding=True, return_tensors="pt").to(self.device)

            with torch.no_grad():
                out_features = self.model(**features)


                embeddings = self.average_pool(out_features.encoder_last_hidden_state, features['attention_mask'])
                embeddings = embeddings.detach()
                if normalize_embeddings:
                    embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)

                # fixes for #522 and #487 to avoid oom problems on gpu with large datasets
                if convert_to_numpy:
                    embeddings = embeddings.cpu()

                all_embeddings.extend(embeddings)

        all_embeddings = [all_embeddings[idx] for idx in np.argsort(length_sorted_idx)]

        if convert_to_tensor:
            all_embeddings = torch.stack(all_embeddings)
        elif convert_to_numpy:
            all_embeddings = np.asarray([emb.numpy() for emb in all_embeddings])

        if input_was_string:
            all_embeddings = all_embeddings[0]

        return all_embeddings

    def _text_length(self, text: Union[List[int], List[List[int]]]):
        """
        Help function to get the length for the input text. Text can be either
        a list of ints (which means a single text as input), or a tuple of list of ints
        (representing several text inputs to the model).
        """

        if isinstance(text, dict):              #{key: value} case
            return len(next(iter(text.values())))
        elif not hasattr(text, '__len__'):      #Object has no len() method
            return 1
        elif len(text) == 0 or isinstance(text[0], int):    #Empty string or list of ints
            return len(text)
        else:
            return sum([len(t) for t in text])      #Sum of length of individual strings
        



class HuggingFaceEmbeddings(BaseModel, Embeddings):
    """HuggingFace sentence_transformers embedding models.

    To use, you should have the ``sentence_transformers`` python package installed.

    Example:
        .. code-block:: python

            from langchain.embeddings import HuggingFaceEmbeddings

            model_name = "sentence-transformers/all-mpnet-base-v2"
            model_kwargs = {'device': 'cpu'}
            encode_kwargs = {'normalize_embeddings': False}
            hf = HuggingFaceEmbeddings(
                model_name=model_name,
                model_kwargs=model_kwargs,
                encode_kwargs=encode_kwargs
            )
    """

    client: Any  #: :meta private:
    model_name: str = ""
    """Model name to use."""
    device: Optional[str] = "cuda"
    """Path to store models. 
    Can be also set by SENTENCE_TRANSFORMERS_HOME environment variable."""
    model_kwargs: Dict[str, Any] = Field(default_factory=dict)
    """Key word arguments to pass to the model."""
    encode_kwargs: Dict[str, Any] = Field(default_factory=dict)
    """Key word arguments to pass when calling the `encode` method of the model."""
    multi_process: bool = False
    """Run encode() on multiple GPUs."""

    def __init__(self, **kwargs: Any):
        """Initialize the sentence_transformer."""
        super().__init__(**kwargs)

        self.client = CustomViEmbeddings(model_name=self.model_name, device=self.device)

    class Config:
        """Configuration for this pydantic object."""

        extra = Extra.forbid

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Compute doc embeddings using a HuggingFace transformer model.

        Args:
            texts: The list of texts to embed.

        Returns:
            List of embeddings, one for each text.
        """
        import sentence_transformers

        texts = list(map(lambda x: x.replace("\n", " "), texts))
        if self.multi_process:
            pool = self.client.start_multi_process_pool()
            embeddings = self.client.encode_multi_process(texts, pool)
            sentence_transformers.SentenceTransformer.stop_multi_process_pool(pool)
        else:
            embeddings = self.client.encode(texts, **self.encode_kwargs)

        return embeddings.tolist()

    def embed_query(self, text: str) -> List[float]:
        """Compute query embeddings using a HuggingFace transformer model.

        Args:
            text: The text to embed.

        Returns:
            Embeddings for the text.
        """
        return self.embed_documents([text])[0]
    
    def tokenc(self, text: str, **kwargs) -> List[int]:
        """
        Tokenize input string
        """
        tokens = self.client.tokenizer(text, return_tensors="pt", **kwargs).input_ids[0]
        return tokens
    
    def tokdec(self, tokens: List[int], **kwargs) -> str:
        string = self.client.tokenizer.decode(tokens, **kwargs)
        return string