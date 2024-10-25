# Load libraries
from utils.load_llm_model import prepare_to_load_model, load_model
from utils.make_prompt import make_prompt
from utils.temps_and_models_for_aug_generation import model_list, temp_list

# This function will set up our apikeys and cache directory correctly for loading the models from HF
# This has to be run before transformers is imported
import os
username = os.getenv('USER')
prepare_to_load_model(username)

from transformers import AutoTokenizer
from langchain import PromptTemplate
import pandas as pd
import argparse
import gc
import torch
import logging
import traceback
import datetime

parser = argparse.ArgumentParser()
parser.add_argument('--top_p', type=float, default=0.95, help='LLM sampling restriction')
parser.add_argument('--max_new_tokens', type=int, default=60, help='max number of new tokens for LLM to make')
parser.add_argument('--min_new_tokens', type=int, default=5, help='min number of new tokens for LLM to make')
parser.add_argument('--num_beams', type=int, default=5, help='Number of beams for beam search')
parser.add_argument('--num_beam_groups', type=int, default=5, help='Number of beam groups for beam search')
parser.add_argument('--repetition_penalty', type=float, default=1, help='Penalizes model repetitiveness')
parser.add_argument('--num_return_sequences', type=int, default=5, help='How many outputs to return for each prompt')
parser.add_argument('--do_sample', action='store_true', default=True, help='Activates sampling rather than greedy decoding')
parser.add_argument('--bad_words', type=list, default=['resources','Resources', '2', '\n'], help='Tokens to forbid model from generating')
parser.add_argument('--torch_use_cuda_dsa', type=list, default=['resources','Resources', '2', '\n'], help='To use cuda dsa')

args = parser.parse_args()


for model_id in model_list:
    print(f'Loading tokenizer for {model_id}')
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    error_count = 0

    for temp in temp_list:

        # Load the model pipeline. After this, 'model' will be a HF pipeline object.
        model = load_model(model_id, 
                           temperature=temp, 
                           top_p=args.top_p, 
                           min_new_tokens=args.min_new_tokens,
                           max_new_tokens=args.max_new_tokens, 
                           num_beams=args.num_beams,
                           num_beam_groups=args.num_beam_groups,
                           repetition_penalty=args.repetition_penalty,
                           num_return_sequences=args.num_return_sequences,
                           do_sample=args.do_sample,
                           tokenizer=tokenizer,
                           bad_words=args.bad_words
                           )

        print(f'Model {model_id} loaded with temperature {temp}. Model device: {model.device}')

        for train_idx in [1,2]:

            # load training data
            train_loc = os.path.join('data','stratified_data_splits',str(train_idx),'train.csv')
            train = pd.read_csv(train_loc)

            # Create data on a loop
            # Initialize some variables that will help us run the loop smoothly
            categories = ['Resource-related', 'Non-resource-related']
            resource_labels = {'Resource-related':1, 'Non-resource-related':0}
            count = 0 
            print('Beginning augmentation loop.')
            while count < 10:
                # Get new augments
                print('Making prompts')
                new_prompts_and_augs = make_prompt(n_in=5, data=train)
                print('Generating augments')
                try:
                    for category in categories:
                        raw_new_augs = model(new_prompts_and_augs[category]['prompt'])
                        new_augs = [aug['generated_text'][len(new_prompts_and_augs[category]['prompt']):].split('\n2.')[0].strip('\n\'" \t') for aug in raw_new_augs]
                        new_prompts_and_augs[category]['augments'] = new_augs
                except Exception as e:
                    current_datetime = datetime.datetime.now()
                    formatted_datetime = current_datetime.strftime("%Y-%m-%d_%H-%M-%S")
                    error_type = type(e).__name__
                    error_count += 1
                    filename = f"{username}_{formatted_datetime}.log"
                    print(f'\nError type: {error_type}. Skipping this round. Error count: {error_count}\n')
                    if error_count == 100:
                        break
                    logging.basicConfig(filename=filename, level=logging.ERROR)
                    logging.error(traceback.format_exc())
                    continue
                # Get path in which augments are stored
                augs_path = os.path.join('data','stratified_data_splits',str(train_idx),'augments.csv')

                # Load existing augments
                print('Loading existing augs')
                try:
                    augs_df = pd.read_csv(augs_path)
                except FileNotFoundError: # If these are the first augs
                    augs_df = pd.DataFrame()

                # Prepare new augments in df with other relevant metadata
                print('Saving new augs')
                for category in categories:
                    new_prompts_and_augs[category]['new_augs_df'] = pd.DataFrame({'text':new_prompts_and_augs[category]['augments'],
                                                                                  'model_id':model_id,
                                                                                  'temperature':temp,
                                                                                  'top_p':args.top_p,
                                                                                  'min_new_tokens':args.min_new_tokens,
                                                                                  'max_new_tokens':args.max_new_tokens,
                                                                                  'num_beams':args.num_beams,
                                                                                  'num_beam_groups':args.num_beam_groups,
                                                                                  'repetition_penalty':args.repetition_penalty,
                                                                                  'do_sample':args.do_sample,
                                                                                  'num_return_sequences':args.num_return_sequences,
                                                                                  'prompt':new_prompts_and_augs[category]['prompt'],
                                                                                  'resources':resource_labels[category],
                                                                                  'bad_words':",".join(args.bad_words)
                                                                                 })


                # Add new augments df to the existing augments
                combined_augs_df = pd.concat([augs_df,
                                             new_prompts_and_augs['Resource-related']['new_augs_df'],
                                             new_prompts_and_augs['Non-resource-related']['new_augs_df']])
                combined_augs_df.reset_index(drop=True, inplace=True)

                # Save the newly increased set of augments
                combined_augs_df.to_csv(augs_path, index=False)

                # Output an update
                count += 1
                print(f'Fold: {train_idx} Total new augments created so far for each category: {count * args.num_return_sequences}')
        # Delete the model and free up GPU memory
        del model
        torch.cuda.empty_cache()

        # Run the garbage collector
        gc.collect()

    # Delete the tokenizer and free up memory
    del tokenizer
    torch.cuda.empty_cache()

    # Run the garbage collector
    gc.collect()
