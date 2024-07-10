# Baby Examples for LLM Inference and Finetuning

## Prepare the Environment
Start with creating a conda environment in python 3.10. This version of python is the most compatible one with current Huggingface Transformers library. 

```
conda create --name baby python=3.10
conda activate baby
```

Then install the required packages.
```
pip install -r requirements.txt
```

## Infer
To run the inference on a pre-trained model, you can use the following command. Huggingface will automatically download the modelif it does not detect it in your local Huggingface cache. So you don't need to download the model manually. 

However, be aware where the huggingface cache is located in your system. I recommend you to manually set this repo every time you run LLMs. Later versions of huggingface uses HF_HOME rather than TRANSFORMERS_CACHE, but only setting HF_HOME could lead to error. I recommend you to set both of them.
```
export HF_HOME=[/path/to/huggingface/cache]
export TRANSFORMERS_CACHE=[/path/to/huggingface/cache]
```

Then you can run the inference script.
```
python infer.py
```

## Finetune