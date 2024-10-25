# Import libraries
import os
import pdb
def prepare_to_load_model(username):
    """
    Set cache directory and load Huggingface api key
    username should be your clemson username. This will be used to set the location of your scratch directory. 
    We need to store models in the scratch directory because the models are too big for the home directory.
    """

    directory_path = os.path.join('/scratch',username)

    # Set Huggingface cache directory to be on scratch drive
    if os.path.exists(directory_path):
        hf_cache_dir = os.path.join(directory_path,'hf_cache')
        if not os.path.exists(hf_cache_dir):
            os.mkdir(hf_cache_dir)
        print(f"Okay, using {hf_cache_dir} for huggingface cache. Models will be stored there.")
        assert os.path.exists(hf_cache_dir)
        os.environ['TRANSFORMERS_CACHE'] = f'/scratch/{username}/hf_cache/'
    else:
        error_message = f"Are you sure you entered your username correctly? I couldn't find a directory {directory_path}."
        raise FileNotFoundError(error_message)

    # Load Huggingface api key
    api_key_loc = os.path.join('/home', username, '.apikeys', 'huggingface_api_key.txt')

    if os.path.exists(api_key_loc):
        print('Huggingface API key loaded.')
    else:
        error_message = f'Huggingface API key not found. You need to get an HF API key from the HF website and store it at {api_key_loc}.\n' \
                        'The API key will let you download models from Huggingface.'
        raise FileNotFoundError(error_message)
        
        
def load_model(model_id, 
               temperature, 
               top_p, 
               min_new_tokens, 
               max_new_tokens,
               num_beams,
               num_beam_groups,
               repetition_penalty,
               do_sample,
               num_return_sequences,
               tokenizer,
               bad_words
              ):
    # Model id should be in the huggingface format <organization>/<modelname>.
    print(f"HF cache: {os.environ['TRANSFORMERS_CACHE']}")
    from transformers import AutoModelForCausalLM, LlamaTokenizer, LlamaForCausalLM, GenerationConfig, pipeline
    
    # Prepare to forbid "bad words"
    bad_words_ids = tokenizer(bad_words, add_special_tokens=True).input_ids

    print(f'Loading model for {model_id}')
    if(model_id == 'mosaicml/mpt-7b-instruct' or model_id == 'mosaicml/mpt-7b'):
        model = AutoModelForCausalLM.from_pretrained(model_id, load_in_8bit=True, device_map='auto', trust_remote_code=True)
    else:
        model = AutoModelForCausalLM.from_pretrained(model_id, load_in_8bit=False)
    model.to('cpu')
    
    print(f'Instantiating pipeline for {model_id}')
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        min_new_tokens=min_new_tokens,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
        do_sample=do_sample,
        num_return_sequences=num_return_sequences,
        bad_words_ids=bad_words_ids
    )
    
    return pipe
