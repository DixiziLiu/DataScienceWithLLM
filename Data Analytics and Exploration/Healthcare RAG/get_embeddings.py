import json
import pandas as pd
import os
from langchain.document_loaders import DataFrameLoader
import sys
from langchain.embeddings import OpenAIEmbeddings # for creating embeddings
from langchain.vectorstores import Chroma # for the vectorization part
from langchain.embeddings import HuggingFaceEmbeddings
        
# Load in data file as command line argument
def load_data_to_dataframe(file_path):
    # Check file extension and load accordingly
    if file_path.endswith('.json'):
        df = pd.read_json(file_path)
    elif file_path.endswith('.csv'):
        df = pd.read_csv(file_path)
    else:
        raise ValueError("Unsupported file format. Please provide a .json or .csv file.")
    
    return df

# Prompt user for a column name and validate it
def get_valid_column_name(df):
    while True:
        column_name = input("Please enter the name of the column that contains the page content: ")
        if column_name in df.columns:
            return column_name
        else:
            print("The column name you entered is not valid. Please try again.")  # Inform the user and ask again

def generate_vector_store(data,
                          cache_loc,
                          embedding_model_name='sentence-transformers/all-mpnet-base-v2'):

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

    # Create embeddings for each eval
    db = FAISS.from_documents(data, hf)

    return db

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

    # Create embeddings for each eval
    db = FAISS.load_local(local_loc, embeddings=hf)

    return db

def generate_eval_embeddings(df, content_col):
    print("Generating embeddings...")
    username = os.environ.get('USER')
    cache_loc = os.path.join('/','scratch',username,'hf_cache')
    
    loader = DataFrameLoader(df, page_content_column=content_col)

    content = loader.load_and_split()
    
    saved_already = False

    if saved_already:
        db = load_saved_vector_store(cache_loc)
    else:
        # Generate embeddings for resident evals
        db_embeddings = generate_vector_store(content, cache_loc)

        db_embeddings.save_local(os.path.join("data","faiss_index"))
        
    return db_embeddings

def main():
    # Check if an argument was provided (1 argument besides the script name)
    if len(sys.argv) < 2:
        print("Usage: script.py <data_file>")
        sys.exit(1)

    data_file = sys.argv[1]  # Get the file location from the command line argument

    print(f"File location provided: {data_file}")
    
    # Create dataframe from passed in data
    df = load_data_to_dataframe(data_file)
    
    content_column = get_valid_column_name(df)

    db_embeddings = generate_eval_embeddings(df, content_column)
    
    print("Embeddings have been successfully generated!")

if __name__ == "__main__":
    main()