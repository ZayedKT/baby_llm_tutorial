# Baby Tutorial for LLM Beginners

This repository provides the simplest examples of using an LLM to do:
- Infer
- Finetune
- How to use .sbatch files for HPC
- RAG (Retrieval Augmented Generation)
- Agent

Which are basically all you need to do with LLM in your projects. As you delve deeper, your code might need to get more complex, but the basic principles are the same. Best of luck on your exploitation to the world of LLMs!

## Prepare the Environment

### Setting up conda environment
Start with creating a conda environment in python 3.10. This version of python is the most compatible one with current Huggingface Transformers library. 

```
conda create --name baby python=3.10
conda activate baby
```

Then install the required packages.
```
pip install -r requirements.txt
```
Note that most important packages I used here are not the latest. The principle of choosing libraries is to pick the version that WORKS. transformers==4.31.0 is by far the most compatible and stable version, and is also welcomed by many researchers. However, if the model you use requires more recent version, you should always upgrade the library.

### Prepare keys
In your projects, a lot of querying will require you to provide keys. For example, OpenAI, Huggingface, etc.. If you are using github, you cannot specifically save those keys specifically. Therefore, I recommend you to save all your keys in a "keys.txt" file in the root directory of all your projects. Then you can read the keys from the file. 

Below is my template for keys.txt:
```
Huggingface: [your huggingface key]
OpenAI: [your openai key]
Claude: [your claude key]
Gemini: [your gemini key]
```

### Setting the cache dir
When you load a model from huggingface (transformers), it will download the models automatically to your local cache: [HUGGINGFACE_CACHE]. This is a system path, and you can view it with `echo $TRANSFORMERS_CACHE`. I recommend that you set it to your own design, and ALWAYS run `export TRANSFORMERS_CACHE=[your local directory for huggingface]` before you run anything.



## Infer
To run the inference on a pre-trained model, you can use the following command. Huggingface will automatically download the modelif it does not detect it in your local Huggingface cache. So you don't need to download the model manually. 

However, be aware where the huggingface cache is located in your system. I recommend you to manually set this repo every time you run LLMs. Later versions of huggingface uses HF_HOME rather than TRANSFORMERS_CACHE, but only setting HF_HOME could lead to error. I recommend you to set both of them.
```
export TRANSFORMERS_CACHE=[/path/to/huggingface/cache]
```

Then you can run the inference script.
```
python infer.py
```


## Finetune (with QLoRA)
"Finetuning" basically means training an already-trained model. However, the training process is usually computation-comsuming, and requires a lot of memory. Therefore, the code I provide here uses QLoRA to finetune the model. 
- "Q" means quantization, where the model's weights are saved in lower precision (e.g., 8-bit integers). In this way, the training will take much less memory, so you can do it with fewer GPUs. 
- "LoRA" is short for low-rank adaptation, which means the model's weights are decomposed into two matrices, and the training is done on these two matrices. This also reduces the memory requirement and computation cost. 
Note that, both methods come with a drop of accuracy when your dataset is large. However, when you have a very small dataset (say less than 1000 entries), then you can always use QLoRA to finetune the model.

## Using sbatch files
If you use NYUAD HPC (aka Jubail), then you will need to use sbatch files to run your code. Long story short, an .sbatch file does two things: let the system know what computing resources you need (gpus, cpus, memory), and run your code. 

You will access your jubail account with:
```
ssh [your netid]@jubail.abudhabi.nyu.edu
```
Then you can submit your job with:
```
sbatch [your sbatch file]
```

To gain access to the NYUAD HPC, you will need to go to the official website and complete the tutorial and quiz. After that, you will submit a form to the department, which will be approved by your professor. If you have any question regarding HPC, always shoot an email to the HPC team. They are very helpful and responsive. 


## RAG (Retrieval Augmented Generation)
(Note this is more related to industrial application than research)

RAG is a model that combines a generative model with a retrieval model. The retrieval model is used to find the most relevant information from a large corpus, and the generative model is used to generate the answer. 

Say you build a server for a company that sells burgers, then you will have a dataset that saves all the ingredients of each burger. You query the language model with a question saying 
```
"Is there cheese in BigMac?"
```
and then the retrieval model will find the most relevant burger (BigMac) and return the ingredients of BigMac. The generative model will then generate the answer based on the ingredients, and say 
```
"Yes, BigMac has cheese"
```
In my example, I provided a .pdf file. Just to give you an idea of how you can use different types of files for your retrieval model. 

Note that the gpt models perform much better than any local model, so I recommend you to use that for learning. All you need to do is to have an OpenAI account, attach a credit card, and get the OpenAI token from its API. 


## Agent
(Note this is more related to industrial application than research)

Sometimes you need the model to use external functions or tools to assist it. For example, when you want to know the weather of Abu Dhabi TODAY, then the model must see the weather report from the internet. This is where the agent comes in. It instructs the model to write code that uses the tools. 

In my example, I provided the code that uses OpenAI model to search on Wikipedia and use calculator. You can ask questions like: What is the size of UAE?
