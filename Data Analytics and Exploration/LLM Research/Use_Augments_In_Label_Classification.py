# Imports
import os
import torch
from datasets import Dataset
from utils.temps_and_models_for_aug_generation import model_list, temp_list, aug_count_list
from scipy.special import softmax
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, roc_auc_score
import json
import pandas as pd
from transformers import TrainingArguments, RobertaTokenizerFast, RobertaForSequenceClassification, EarlyStoppingCallback, Trainer, XLNetForSequenceClassification, XLNetTokenizerFast, DistilBertForSequenceClassification, DistilBertTokenizerFast
import datetime
import numpy as np
import time

# Seed the random number generator with the current time
np.random.seed(int(time.time()))

# Set seed for PyTorch
torch.manual_seed(int(time.time()))

# Get current datetime to format logs and performance evaluation csvs
current_datetime = datetime.datetime.now()
formatted_datetime = current_datetime.strftime("%Y-%m-%d_%H-%M-%S")
filename = f"downstream_classifier_perfs_{formatted_datetime}.csv"

# Set env vars and file locs
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
perfs_csv_loc = os.path.join('data','downstream_classifier_perfs',filename)

# Load the classifier training settings from the JSON file
with open('classifier_training_config.json') as f:
    training_args_dict = json.load(f)
    user = os.getenv('USER')
    output_dir = os.path.join('/scratch',user,formatted_datetime)
    os.mkdir(output_dir)
    training_args_dict['output_dir'] = output_dir
    
def load_real_retips_data(fold:int):
    """
    Given a training fold, load the real RETIPS labeled data. 
    Returns a pd.DataFrame for each of the train and test set.
    
    Args:
        fold (int): Which training fold to use.
        
    Returns:
        tuple(pd.DataFrame, pd.Dataframe): The training data, and the test data.
    """
    
    train_file_path = f'data/stratified_data_splits/{fold}/train.csv'
    test_file_path = f'data/stratified_data_splits/{fold}/test.csv'

    train_data = pd.read_csv(train_file_path)
    test_data = pd.read_csv(test_file_path)
    
    return train_data, test_data
    
def load_augments(fold:int):
    """
    Given a training fold, load the augments based on that fold.
    Returns a pd.DataFrame of the augments.
    
    Args:
        fold (int): Which training fold to use.
        
    Returns:
        pd.DataFrame: The augments.
    """
    
    augments_file_path = f'data/stratified_data_splits/{fold}/augments.csv'
   
    augments = pd.read_csv(augments_file_path)
    
    return augments


def filter_augments(augments:pd.DataFrame, llm_type:str, temp:float):
    """
    Given a pd.DataFrame of augments, a specified LLM type and a temperature, 
    filter the pd.DataFrame to keep only the rows for that LLM type and temperature.
    
    Args:
        augments (pd.DataFrame): The full set of unfiltered augments.
        llm_type (str): The LLM model type we want to keep.
        temp (float): The temperature we want to keep.
        
    Returns:
        pd.DataFrame: The filtered augments.
    """
    
    filtered_aug = augments[(augments['model_id'] == llm_type) & (augments['temperature'] == temp)]
    
    return filtered_aug
            

def combine_real_with_augs(real_data:pd.DataFrame, augments:pd.DataFrame, n_augs_to_sample:int):
    """
    Given some real data and augments, combine them to form a single data frame to be used for training.
    The combined data frame does NOT include all the augments.
    Instead, this function randomly samples from the augments just enough rows 
    in order to balance the real data, so that the final dataframe has the same number of rows in each
    category, before adding more augments equally from each label category.
    
    Note this function will throw an error if there aren't enough augs to satisfy the requested number.
    
    Args:
        real_data (pd.DataFrame): The real data.
        augments (pd.DataFrame): The augments.
        n_augs_to_sample (int): The number of augments to add to the real data.
        
    Returns:
        pd.DataFrame: the combined dataframe.
    """
    # Make the augments col names same as real data
    augments = augments.rename({'resources': 'label', 'text': 'Response'}, axis = 1)
    
    # Get count of real data cases that have and lack the resources label
    non_resource_count = (real_data['label']==0).sum()
    resource_count = (real_data['label']==1).sum()
    
    # Find the imbalance. How many more non_resource cases do we have than resource cases?
    count = non_resource_count - resource_count
    
    # Number of rows to sample
    n_res_augs = min(n_augs_to_sample, count) # Number of augs to sample with only resources label (to balance real data)
    n_eq_augs = max(n_augs_to_sample-count, 0) # Number of augs to sample equally from resources, non-resources
    
    # Create sub-DataFrames that have only one label each
    label_1 = augments[augments['label'] == 1]
    label_0 = augments[augments['label'] == 0]

    # Sample n_res_augs from label_1
    res_sample = label_1.sample(n=n_res_augs, replace=False)

    # Remove the rows that have already been sampled
    label_1 = label_1.drop(res_sample.index)

    # Sample n_eq_augs//2 from the remaining label_1 and from label_0
    eq_sample_label_1 = label_1.sample(n=n_eq_augs // 2, replace=False)
    eq_sample_label_0 = label_0.sample(n=n_eq_augs // 2, replace=False)

    # Concatenate the two equal samples
    eq_sample = pd.concat([res_sample, eq_sample_label_1, eq_sample_label_0])

    # Concatenate to the real data
    df_combined = pd.concat([real_data, eq_sample], ignore_index=True)
        
    return df_combined

    
def dataset_loader(train_df:pd.DataFrame, test_df:pd.DataFrame):
    """
    Given train and test dataframes, put them into pytorch dataloaders.
    
    Args:
        train_df (pd.DataFrame): Training df
        test_df (pd.DataFrame): Test df
        
    Returns:
        tuple(dataloader, dataloader)
    """
    
    train_dataset = Dataset.from_pandas(train_df)
    test_dataset = Dataset.from_pandas(test_df)
    
    return train_dataset, test_dataset


def load_classifier_and_tokenizer(classifier_type:str):
    """
    Given a classifier type, load (from Hugging Face) that model and its tokenizer, for use as a classifier.
    
    Args:
        classifier_type (str): A string specifying which model type to load as classifier.
        
    Returns:
        tuple(model, tokenizer): The classifier and its tokenizer.
    """
    # Load classifier
    num_labels = 2 # Assumes binary classification
    if classifier_type == 'roberta':
        model = RobertaForSequenceClassification.from_pretrained('roberta-base', num_labels=num_labels)
        tokenizer = RobertaTokenizerFast.from_pretrained('roberta-base', max_length = 512)
    elif classifier_type == 'xlnet':
        model = XLNetForSequenceClassification.from_pretrained('xlnet-base-cased', num_labels=num_labels)
        tokenizer = XLNetTokenizerFast.from_pretrained('xlnet-base-cased', max_length = 512)
    elif classifier_type == 'distilbert':
        model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=num_labels)
        tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased', max_length = 512)
    else:
        raise ValueError(f"Invalid model_type: {classifier_type}. Expected 'roberta', 'xlnet', or 'distilbert'.")
    
    return model, tokenizer

def compute_metrics(pred, average = 'binary'):
        """
        Compute custom evaluation metrics for the model.

        Args:
            pred (EvalPrediction): An object that contains the model's predictions and labels for evaluation.

        Returns:
            dict: A dictionary containing the computed metrics.
        """
        # Extract ground truth labels and predicted labels from EvalPrediction object
        labels = pred.label_ids
        preds = pred.predictions.argmax(-1)

        # Compute predicted probabilities using softmax activation along axis 1
        probs = softmax(pred.predictions, axis=1)

        # Compute precision, recall, F1-score, and support using sklearn's precision_recall_fscore_support function
        # Set the 'average' parameter to 'macro' to compute macro-averaged metrics for multi-class classification
        precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average = average)

        # Compute accuracy using sklearn's accuracy_score function
        acc = accuracy_score(labels, preds)

        # Compute area under the ROC curve (AUC) using sklearn's roc_auc_score function
        auc = roc_auc_score(labels, probs[:, 1])

        # Return the computed metrics as a dictionary
        return {
            'accuracy': acc,
            'f1': f1,
            'precision': precision,
            'recall': recall,
            'auc': auc
        }

# Loop over a range of training folds, temps and model_ids, and get performances of resulting models. 
# Save performances in csv.
llm_types = model_list
temps = temp_list
folds = [1,2,3,4]
for fold in folds:
    train_df, test_df = load_real_retips_data(fold)
    all_augments = load_augments(fold)
    classifiers = ['roberta', 'xlnet', 'distilbert']
    
    for llm_type in llm_types:
        for classifier in classifiers:
            for temp in temps:
                for aug_count in aug_count_list:
                    
                    # Set seed for numpy and PyTorch based on current time
                    current_time = int(time.time())
                    np.random.seed(current_time)
                    torch.manual_seed(current_time)
                    
                    # Enable non-deterministic algorithms for GPU
                    torch.backends.cudnn.deterministic = False
                    torch.backends.cudnn.benchmark = True

                    # Make an augmented training dataframe. If we don't have enough augments, skip the rest of aug_count_list
                    try:
                        filtered_augments = filter_augments(augments=all_augments, llm_type=llm_type, temp=temp)
                        augmented_train_data = combine_real_with_augs(real_data=train_df, augments=filtered_augments, n_augs_to_sample=aug_count)
                    except ValueError:
                        continue

                    # Instantiate dict which will store the classifier performance results
                    performance_dict = {'temperature':temp,
                                        'training_fold':fold,
                                        'llm_type':llm_type,
                                        'aug_count':aug_count,
                                        'classifier':classifier}


                    # Create dataset objects for training
                    train_data, test_data = dataset_loader(train_df=augmented_train_data, test_df=test_df)

                    # Load classifier and tokenizer
                    model, tokenizer = load_classifier_and_tokenizer(classifier)

                    # Define a function that will use the tokenizer to tokenize the data, 
                    # and will return the relevant inputs for the model
                    def tokenization(batched_text):
                        return tokenizer(batched_text['Response'], padding = True, truncation=True)

                    train_data = train_data.map(tokenization, batched = True, batch_size = len(train_data))
                    test_data = test_data.map(tokenization, batched = True, batch_size = len(test_data))
                    train_data.set_format('torch', columns=['input_ids', 'attention_mask', 'label'])
                    test_data.set_format('torch', columns=['input_ids', 'attention_mask', 'label'])

                    # Create the TrainingArguments
                    training_args = TrainingArguments(**training_args_dict)

                    # Instantiate the EarlyStoppingCallback to add early stopping based on auc_1
                    early_stopping_callback = EarlyStoppingCallback(
                        early_stopping_patience=20,  # Number of epochs to wait for improvement
                        early_stopping_threshold=0,  # Minimum improvement required to consider as improvement
                    )
                    callbacks = [early_stopping_callback]

                    # instantiate the trainer class and check for available devices

                    trainer = Trainer(
                        model=model,
                        args=training_args,
                        train_dataset=train_data,
                        compute_metrics=compute_metrics,
                        eval_dataset=test_data,
                        callbacks=callbacks
                    )
                    device = 'cuda' if torch.cuda.is_available() else 'cpu'

                    # Perform fine-tuning
                    trainer.train()

                    # Record final model performance (at best epoch)
                    performance_dict.update(trainer.evaluate())

                    # Delete the trained model (otherwise storage will fill)
                    del trainer.model
                    del trainer
                    
                    # Convert the performance_dict to a df
                    perf_df = pd.DataFrame({k: [v] for k, v in performance_dict.items()})

                    # Check if the file already exists
                    file_exists = os.path.isfile(perfs_csv_loc)

                    # Append to file (write header only if the file doesn't exist)
                    perf_df.to_csv(perfs_csv_loc, mode='a', header=not file_exists, index=False)