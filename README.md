# Baby Tutorial for LLM and HPC Beginners

If you are following this tutorial, please `fork` this repo and clone it to your local computer / HPC scratch folder.

This repository provides the simplest examples of using an LLM to do:
- Infer
- Finetune
- Submit jobs with .sbatch files for HPC
- RAG (Retrieval Augmented Generation)
- Agent

Which are basically all you need to do with LLM in your projects. As you delve deeper, your code might need to get more complex, but the basic principles are the same. 

However, please get ready to spend a lot of time preparing the environment. It is the most crucial part, and is usually not trivial for beginners. This repo gives you the most standard way of setting up a virtual environment for ML projects. You are recommended to follow the steps for any programming tasks in the future. 

Note that this tutorial assumes you are using an HPC server (ie. Jubail HPC from NYUAD), so we will provide HPC basics in the tutorial. The procedure is similar if you are using your own computer, except that you don't need to use .sbatch files.

Best of luck on your exploitation to the world of LLMs!




## Prepare the Environment

### Install Miniconda
Miniconda is a package manager that helps you manage different virtual python environments. Normally you would have multiple projects running on your computer, and each project might require different versions of python and packages. Therefore, it is crucial to use a package manager to manage your environments. 

`conda` is by far the simplest way of managing environments, though not the most efficient in terms of both storage and speed. However, for beginners, it is the most user-friendly one, as it is very hard to mess up your system with conda. In the worst case, you can always remove an environment, or uninstall conda itself. However, we don't need all features from Anaconda, so we will just install Miniconda instead. 

To begin with, let's install miniconda by following [this tutorial](https://www.anaconda.com/docs/getting-started/miniconda/install#linux-terminal-installer). 

Two important notes:
- Do this in your /scratch folder. EVERYTHING should be in scratch folder. The only exception is the `.bashrc` file in your home directory, which contains the commands run atomatically every time you log in or call `source ~/.bashrc`.
- Please read the installation logs carefully, as towards the end it will instruct you on how to set conda to the PATH permanently in `.bashrc`. Do not skip this, otherwise your system cannot find `conda` as a command. Once you are done, exit HPC and log in again.


### Setting up a virtual environment with conda
Start with creating a conda environment in python 3.10. This version of python is the most compatible one with current Huggingface Transformers library. 

```
conda create --name baby python=3.10
conda activate baby
```

### Installing required packages
Note that the major machine learning library we will use - pytorch - is dependent on the CUDA version of your system. The current HPC nodes use CUDA 12.2. So we install pytorch-related libraries separately with the following command:
```
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121
```

And then we install the remaining required packages from requirements.txt.
```
pip install -r requirements.txt
```
Note that most important packages I used here are quite old. The principle of choosing libraries is to pick the version that **WORKS**. However, if the model you use requires more recent version, you should always update the library.

If you are running it on your HPC server, you may find that this procedure takes very long time. This is because the default CPU assigned to you at logins is very slow. You may check out a better cpu with a `srun` command as you delve deeper... For now, let's just wait :)



### Prepare keys
In your projects, a lot of querying will require you to provide your token/key. For example, OpenAI, Huggingface, etc.. They verify that you are you, so they can charge you for commercial services, or ban you for private repos. These tokens/keys are generated MANUALLY on respective APIs. For example, if you want to use OpenAI API, you need to create an account on OpenAI, attach a credit card, and then google "How to get an OpenAI API token".

Please understand that these tokens/keys are very private, and you should NEVER share them with anyone. If I steal your tokens/keys, I can use them to do anything on your behalf, and you will be charged for all the usage. Therefore, github does not allow you to push any key, as they would automatically detect them. 

The most standard way of managing keys is to store them as `export` commands in your `.bashrc` file. For example, if you have an OPEN API key, you can add the following line to your `.bashrc` file:
```
export OPENAI_API_KEY=[your openai api key]
```
It doesn't matter how you name it. You may call it `WHAT_an_AMAZING_key` if you want to, as long as you can remember it. We will retrieve it with `os.getenv` in python scripts. 

Similarly, you may also set the following environment variables. But they are optional. 
```
HUGGINGFACE_TOKEN (for some huggingface models with limited access)
DEEPSEEK_API_KEY
PERSPECTIVE_API_KEY
ANTHROPIC_API_KEY
GOOGLE_API_KEY (for gemini api)
```



### Setting the cache dir
When you load a model from huggingface (via transformers library), it will download the models automatically to your local directory `TRANSFORMERS_CACHE`. This is a system path, and you can view it with `echo $TRANSFORMERS_CACHE`. I recommend that you set it to your own design, and ALWAYS run `export TRANSFORMERS_CACHE=[your local directory for huggingface]` before you run anything. Also set `HF_HOME` to the same path, in the same fashion. 



## HPC Commands
If you are doing research/work related to machine learning, most likely you will use a remote server with GPUs. NYUAD Jubail HPC is one of them. It is very well-established and not crowded. There are 2 ways you can use it. I provide both of them in this section.

### Gain Access to Jubail
To gain access to Jubail, you will need to follow [this webpage](https://crc-docs.abudhabi.nyu.edu/hpc/hpc.html) and register an account by submitting an applicaion form. The "Faculty Sponsor" should be the professor of the lab you are working for. 

Once you have your account approved, you will be able to access your jubail account with:
```
ssh [your netid]@jubail.abudhabi.nyu.edu
```

When you log in, you will see your disk quota for 3 directories:
```
/home/[Your NetID]
/scratch/[Your NetID]
/archive/[Your NetID]
```
The home directory will be the default path to all of your downloads and installations. However, it has only 50G. The archive folder saves files that you don't normally use but want to store for history. So please make sure whatever project you work on, and whatever environment/model you download, always use your scratch folder. You should install miniconda and repeat the above steps inside your scratch folder.

If you have any question regarding HPC, always shoot an email to the HPC team. They are very helpful and responsive. 

### Submit a Job with a .sbatch File
Then you can submit your job with:
```
sbatch [your sbatch file]
```
If you want a session to only run one file for a long time (10 hours), then creating a .sbatch file is the best choice. A .sbatch file does two things: let the system know what computing resources you need (gpus, cpus, memory), and run your code. You can find a .sbatch file in the infer foler. Please read it carefully and make sure you understand each line. 


### Create an Interactive Session with Computing Resource
There is another way to check out computing resource from HPC - initializing an interactive session with `srun`. This is useful when you want to do some quick experiments.
```
srun --partition=nvidia --constraint=80g --nodes=1 --ntasks=1 --cpus-per-task=1 --gres=gpu:a100:1 --time=71:59:59 --mem=10GB --pty /bin/bash
```
Note you can replace $USER with your netID. 
```
watch -n -1 squeue -u $USER
```
Then Ctrl+C to come out of the watch session, and ssh to the interactivate session. (ie. "ssh cn025")
```
ssh [NODELIST]
```



### VSCode GUI (But not as you think!)
Everybody would want to be able to edit code in a GUI! However, you cannot remote-control your HPC server with local VSCode, as we have already mentioned, the default cpu is very slow. Connecting to the server with VSCode will most likely lead to timeout or unstable connections. However, our diligent HPC team has developed a web-based VSCode GUI that you can use.

Log in via [this link](https://ood.hpc.abudhabi.nyu.edu/pun/sys/dashboard/) with your NYU account. 
1. Select "Interactive Apps" on the top panel
2. Scroll way down and select "Code Server"
3. Set your Num of hours and "Working directory" to /scratch/[your netid]/. And then launch.

You will have a session with VSCode GUI real quick! Note that this API only supports a very limited number of third-party packages, so if you use auto-completion (ie. Github Copilot) on a daily basis, you cannot use it here unfortunately. 




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
