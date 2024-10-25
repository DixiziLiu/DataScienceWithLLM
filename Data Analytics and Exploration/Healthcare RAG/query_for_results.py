import os
cache_loc = "/scratch1/alshelt/hf" # Change path to where you want the model to be stored
os.environ["HF_HOME"] = cache_loc

from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer

def read_prompt():
    with open('rag_prompt.txt', 'r') as file:
        prompt = file.read()
        
    return prompt

def main():
    prompt = read_prompt()

    model_name_or_path = "TheBloke/Llama-2-70B-chat-AWQ"

    # Load model
    model = AutoAWQForCausalLM.from_quantized(model_name_or_path, fuse_layers=True,
                                              trust_remote_code=False, safetensors=True).to("cuda")
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=False)
    
    print("\n\n*** Generate:")

    tokens = tokenizer(
        prompt,
        return_tensors='pt'
    ).input_ids.cuda()

    # Generate output
    generation_output = model.generate(
        tokens,
        do_sample=True,
        temperature=0.7,
        top_p=0.95,
        top_k=40,
        max_new_tokens=512
    )

    output = tokenizer.decode(generation_output[0])
    
    # Get rid of prompt in output (only show generated response)
    cleaned_output = output.split('[/INST] ')[-1]
    cleaned_output = cleaned_output.strip()
    print(cleaned_output)

if __name__ == "__main__":
    main()