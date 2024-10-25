import sys
import pandas as pd
import os
from langchain.embeddings import HuggingFaceEmbeddings

def read_file(file_path):
    # Determine the file type and read accordingly
    if file_path.endswith('.csv'):
        return pd.read_csv(file_path)
    elif file_path.endswith('.json'):
        return pd.read_json(file_path)
    else:
        print("Unsupported file format. Please provide a CSV or JSON file.")
        return None
    
def process_relevance_query(df):
    column = input("Enter the name of the column with the relevant information for generating similarity scores: ")
    
    # Get list of data from chosen column
    return df[column].tolist()
    
def get_relevance_query():
    # Check if a command line argument is provided
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
        dataframe = read_file(file_path)
        if dataframe is not None:
            print("File successfully read into DataFrame.")
            
            return dataframe
        else:
            print("Failed to read the file into DataFrame.")
    else:
        # Prompt the user for a relevance query if no command line argument is provided
        relevance_query = input("No file provided. Please enter a relevance query: ")
        
        return relevance_query
    
def load_saved_vector_store(cache_loc,
                            embedding_model_name='sentence-transformers/all-mpnet-base-v2',
                            local_loc=os.path.join('data','faiss_index'),
                           ):

    # Load HF model that will be used to create the embeddings
    hf = HuggingFaceEmbeddings(model_name=embedding_model_name)
    model_kwargs = {'device': 'cuda'}
    encode_kwargs = {'normalize_embeddings': False}
    hf = HuggingFaceEmbeddings(
        model_name=embedding_model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs,
        cache_folder = cache_loc
    )

    # Load the vector store that will be used to store/search the embeddings
    from langchain.vectorstores import FAISS

    db = FAISS.load_local(local_loc, embeddings=hf)

    return db

def get_size_of_vector_db(db_embedding):
    num_documents = len(db_embedding.index_to_docstore_id)

    return num_documents

def generate_similarity_scores():
    # Load in vector store
    username = os.environ.get('USER')
    cache_loc = os.path.join('/','scratch',username,'hf_cache')
    db_embedding = load_saved_vector_store(cache_loc)

    relevance_query = get_relevance_query()

    if isinstance(relevance_query, pd.DataFrame):
        relevance_data = process_relevance_query(relevance_query)

    else:
        relevance_data = [relevance_query]

    columns = ['text', 'score']
    scores_df = pd.DataFrame(columns=columns)

    # Get size of vector database
    num_documents = get_size_of_vector_db(db_embedding)
    
    print("Generating similarity scores...")

    for i in range(len(relevance_data)):
        query_docs_res = db_embedding.similarity_search_with_score(relevance_data[i], k=num_documents)

        for j in range(num_documents):
            # Add new row
            new_row = pd.DataFrame([[query_docs_res[j][0].page_content, query_docs_res[j][1]]], columns=columns)
            scores_df = pd.concat([scores_df, new_row], ignore_index=True)

    return scores_df

def create_min_scores_df(scores_df):
    # Ensure scores are numeric for idxmin()
    scores_df['score'] = pd.to_numeric(scores_df['score'])

    # Group cols by 'score'
    group_cols = [col for col in scores_df.columns if col != 'score']
    grouped = scores_df.groupby(group_cols)

    # Find index of lowest score for each group
    min_score_idx = grouped['score'].idxmin()

    # Filter df to only keep cols with lowest scores
    result_df = scores_df.loc[min_score_idx]

    # Reset the index of the result DataFrame
    result_df = result_df.reset_index(drop=True)

    return result_df

def prepare_data_for_prompt(result_df):
    # Gather n most relevant examples for prompt
    n = int(input("Enter the number of relevant examples to have in the prompt: "))
    relevant_df = result_df.sort_values(by='score', ascending=True).head(n)
    
    return relevant_df

def define_system_prompt():
    sys_prompt = input("Enter the system prompt: ")
    return sys_prompt

def define_query_prompt():
    query_prompt = input("Enter the prompt that will be used to query the LLM: ")
    return query_prompt

def get_examples(df):
    evals = []
    example_num = 1
    for index, row in df.iterrows():
        if row['text'] == 'na':
            continue
        else:
            eval = "ex. " + str(example_num) + ": " + row['text']
            evals.append(eval)
            example_num += 1
        
    return evals

def create_prompt(sys_prompt, query_prompt, relevant_df):
    
    relevant_examples = get_examples(relevant_df)
    
    prompt_text = """\
<s>[INST] <<SYS>> \
{sys_prompt}
<</SYS>> \

{query_prompt}

Each example is denoted by the 'ex. n: ' prefix, where 'n' is \
the number of the example. \
Please provide explicit citations for each example \
in the prompt that you reference. \
Reference the citation numbers in the text of the response where appropriate \
to provide in-text citations for the examples in the prompt (e.g. Example sentence here [ex. 1]). \
You should prioritize referencing the provided \
examples in the prompt to formulate your response. \

### BEGIN EXAMPLES \
{relevant_examples} \
### END EXAMPLES \

[/INST] \
"""
    prompt_text = prompt_text.format(sys_prompt=sys_prompt, query_prompt=query_prompt, relevant_examples=relevant_examples)
    
    return prompt_text

def main():
    sys_prompt = define_system_prompt()
    query_prompt = define_query_prompt()
    
    username = os.environ.get('USER')
    cache_loc = os.path.join('/','scratch',username,'hf_cache')
    db_embedding = load_saved_vector_store(cache_loc)

    scores_df = generate_similarity_scores()
    
    result_df = create_min_scores_df(scores_df)

    relevant_df = prepare_data_for_prompt(result_df)
    
    prompt = create_prompt(sys_prompt, query_prompt, relevant_df)
    
    # Save prompt in text file
    with open('rag_prompt.txt', 'w') as file:
        file.write(prompt)
    
    print("Prompt has been created! Check the 'rag_prompt.txt' file that has been generated to review/edit it.")

if __name__ == "__main__":
    main()