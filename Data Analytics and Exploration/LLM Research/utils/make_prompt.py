# Imports
from langchain import PromptTemplate
import os

# Import template text
file_path = os.path.join('utils','prompt_template.txt')

with open(file_path, 'r') as file:
    template = file.read()

# Now the variable `template` contains the contents of prompt_template.txt as a string

def make_prompt(n_in, 
                data, 
                template=template):
    
    # Separate the data into two dfs by label
    df0 = data[data.label==0]
    df1 = data[data.label==1]
    
    # Pull n_in responses at random out of each new df
    non_resource_examples = df0['Response'].sample(n=n_in, replace=False)
    resource_examples = df1['Response'].sample(n=n_in, replace=False)
    non_resource_example_text = ''
    resource_example_text = ''
    for i,ex in enumerate(non_resource_examples): non_resource_example_text += f'\n{i+1}. {ex}'
    for i,ex in enumerate(resource_examples): resource_example_text += f'\n{i+1}. {ex}'
    
    # Construct the final prompt
    prompt_template = PromptTemplate(template=template, 
                                     input_variables=["category", 
                                                      "other_category",
                                                      "category_examples", 
                                                      "other_category_examples"])
    
    # Integrate randomly selected data points into prompt templates for each category
    res_prompt_input_dict = {'category':'Resource-related',
                             'other_category':'Non-resource-related',
                             'category_examples':resource_example_text,
                             'other_category_examples':non_resource_example_text}
    non_res_prompt_input_dict = {'category':'Non-resource-related',
                                 'other_category':'Resource-related',
                                 'category_examples':non_resource_example_text,
                                 'other_category_examples':resource_example_text}
    
    # Get one version of the prompt to extract resource augs, one for non-resource augs
    res_prompt = prompt_template.format(**res_prompt_input_dict)
    non_res_prompt = prompt_template.format(**non_res_prompt_input_dict)
    
    # Package them in a dict
    prompts = {'Resource-related':{'prompt':res_prompt},
               'Non-resource-related':{'prompt':non_res_prompt}}
    
    return prompts