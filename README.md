

# Sun-Assistant





## Installation
To load models in 4bits with transformers and bitsandbytes, you have to install accelerate and transformers from source and make sure you have the latest version of the bitsandbytes library (0.39.0). After installing PyTorch (follow instructions [here](https://pytorch.org/get-started/locally/)), you can achieve the above with the following command:
```bash
pip install -U -r requirements.txt
```

## Getting Started
The `qlora.py` code is a starting point for finetuning and inference on various datasets.
Basic command for finetuning a baseline model on the Alpaca dataset:
```bash
python qlora.py --model_name_or_path <path_or_name>
```



### Tutorials and Demonstrations

You can host your own gradio Guanaco demo directly in Colab following [this notebook](https://colab.research.google.com/drive/17XEqL1JcmVWjHkT-WczdYkJlNINacwG7?usp=sharing). 
In addition, here are Colab notebooks with examples for inference and finetuning using QLoRA:
- [Inference notebook](https://colab.research.google.com/drive/1ge2F1QSK8Q7h0hn3YKuBCOAS0bK8E0wf?usp=sharing)
- [Finetuning notebook](https://colab.research.google.com/drive/1VoYNfYDKcKRQRor98Zbf2-9VQTtGJ24k?usp=sharing)


### Quantization
Quantization in fine-tuning phase are controlled from the `BitsandbytesConfig` ([see HF documenation](https://huggingface.co/docs/transformers/main_classes/quantization#transformers.BitsAndBytesConfig)) as follows:
- Loading in 4 bits is activated through `load_in_4bit`
- The datatype used for the linear layer computations with `bnb_4bit_compute_dtype`
- Nested quantization is activated through `bnb_4bit_use_double_quant`
- The datatype used for qunatization is specified with `bnb_4bit_quant_type`. Note that there are two supported quantization datatypes `fp4` (four bit float) and `nf4` (normal four bit float). The latter is theoretically optimal for normally distributed weights and we recommend using `nf4`.

```python
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path='/name/or/path/to/your/model',
        load_in_4bit=True,
        device_map='auto',
        max_memory=max_memory,
        torch_dtype=torch.bfloat16,
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type='nf4'
        ),
    )
```

### Paged Optimizer
You can access the paged optimizer with the argument `--optim paged_adamw_32bit`

### Guanaco Finetuning
You can select `--dataset oasst1` to load the OpenAssistant dataset that was used to train Guanaco. You can also find it on HF at [timdettmers/openassistant-guanaco](https://huggingface.co/datasets/timdettmers/openassistant-guanaco).

We include scripts to reproduce the hyperparameters of Guanaco model training for various sizes at `./scripts/finetune_guanaco*.sh`. Make sure to (1) edit the `model_name_or_path` to your LLaMA checkpoint (2) adjust `per_device_train_batch_size` and `gradient_accumulation_steps` so that they multiply to 16 and fit on your device. 

### Using Local Datasets

You can specify the path to your dataset using the `--dataset` argument. If the `--dataset_format` argument is not set, it will default to the Alpaca format. Here are a few examples:

- Training with an *alpaca* format dataset:
  ```bash
  python qlora.py --dataset="path/to/your/dataset"
  ```
- Training with a *self-instruct* format dataset:
   ```bash
   python qlora.py --dataset="path/to/your/dataset" --dataset_format="self-instruct"
   ```

### Multi GPU
Multi GPU training and inference work out-of-the-box with Hugging Face's Accelerate. Note that the `per_device_train_batch_size` and `per_device_eval_batch_size` arguments are  global batch sizes unlike what their name suggest.

When loading a model for training or inference on multiple GPUs you should pass something like the following to `AutoModelForCausalLM.from_pretrained()`:
```python
device_map = "auto"
max_memory = {i: '46000MB' for i in range(torch.cuda.device_count())}
```



## Known Issues and Limitations
Here a list of known issues and bugs. If your issue is not reported here, please open a new issue and describe the problem.

1. 4-bit inference is slow. Currently, our 4-bit inference implementation is not yet integrated with the 4-bit matrix multiplication
2. Resuming a LoRA training run with the Trainer currently not supported by HF.
3. Currently, using `bnb_4bit_compute_type='fp16'` can lead to instabilities. For 7B LLaMA, only 80% of finetuning runs complete without error. We have solutions, but they are not integrated yet into bitsandbytes.
4. Make sure that `tokenizer.bos_token_id = 1` to avoid generation issues.
5. If you get an this [issue](https://github.com/artidoro/qlora/issues/82) ("illegal memory access") then you should use a newer HF LLaMA conversion or downgrade your PyTorch version.
 



## Citation

```bibtex
@article{dettmers2023qlora,
  title={QLoRA: Efficient Finetuning of Quantized LLMs},
  author={Dettmers, Tim and Pagnoni, Artidoro and Holtzman, Ari and Zettlemoyer, Luke},
  journal={arXiv preprint arXiv:2305.14314},
  year={2023}
}
```


