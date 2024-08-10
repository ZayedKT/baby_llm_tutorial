""" Boyuan:
Hi all. This is an example inference code written by me. This script downloads the 
model checkpoint (in this case facebook/opt-125m, one of the smallest language models 
on huggingface), and generates the output. Since the model is very small, the inference 
should be very fast (finishing in around a second). Feel free to explore other models on 
huggingface, and replace the model_name variable with your chosen one.
"""

from transformers import AutoTokenizer, AutoModelForCausalLM
import time
import torch


def main():
    # Initialize model. 
    model_name = "facebook/opt-125m"
    tokenizer = AutoTokenizer.from_pretrained(model_name, device_map="auto", use_fast=False)
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")
    
    # Initialize the prompt. The model will continue writing from the existing prompt
    prompt = "Question: What is the weather like in Abu Dhabi?\nAnswer: "

    # Generate the output from the prompt. Take some time to understand what each parameter in .generate() means
    input_ids = tokenizer.batch_encode_plus([prompt], return_tensors="pt").to(torch.cuda.current_device())
    start_generating = time.time()
    with torch.no_grad():
        generated_ids = model.generate(
            **input_ids,
            use_cache = True,
            pad_token_id = tokenizer.eos_token_id,
            max_new_tokens = 300,
            do_sample = True,
            temperature = 0.8,
            top_p = 0.9,
        )
    print(f"Time to generate the solution was {(time.time()-start_generating):.2f} seconds")
    answer_ids = generated_ids[:, len(input_ids['input_ids'][0]):]
    answer_text = tokenizer.batch_decode(answer_ids, skip_special_tokens=False)
    print(f"Generated output is:\n{answer_text[0]}")      
                

if __name__== "__main__":
    main()