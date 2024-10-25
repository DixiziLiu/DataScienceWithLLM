# Hospital Staff Narratives and Incident Reports RAG Framework for Healthcare Analytics

This repository provides a Retrieval-Augmented Generation (RAG) framework specifically designed for healthcare incident reporting and operational analysis. The framework retrieves relevant documents from large healthcare datasets to generate prompts for Large Language Models (LLMs), enabling hospital managers and healthcare professionals to uncover actionable insights for improving patient safety and operational efficiency.

## Project Overview
This RAG framework was developed as part of a project exploring over 500 narratives and 70 million healthcare incident reports, using AI to optimize decision-making in patient safety and operations. By integrating LLM-based retrieval, the solution enhances how healthcare organizations analyze everyday challenges, uncovering hidden themes to support better adaptation and resilience in hospital settings.

## Key Features:
Optimized Retrieval: Leverages embeddings to extract the most relevant documents from vast incident report datasets, enabling focused queries that address specific hospital safety concerns.
LLM Prompt Engineering: Employs a fine-tuned Llama 3-70B model and other large-scale models to support retrieval-based generative responses.
Actionable Insights: Transforms complex healthcare data into understandable insights for hospital administrators, enhancing data-driven and AI-supported decision-making.
Getting Started
Given the computational resources required for large-scale text processing and model inference, it is recommended to clone this repository onto a high-performance computing cluster such as Palmetto.

## Core Scripts:
**get_embeddings.py**: Extracts vector embeddings from a specified dataset (in CSV or JSON format), facilitating the retrieval of relevant documents based on semantic similarity. Use this script with "python3 get_embeddings.py <dataset_path>" and specify the content column relevant to your query.

**make_a_prompt.py**: Generates a retrieval-augmented prompt based on the extracted embeddings, ready for LLM input. Optionally, you can provide a dataset to tailor the relevance query or manually input the query text. Run it with "python3 create_prompt.py <optional_dataset_path>".

**query_for_results.py**: Queries an LLM using the generated prompt to retrieve high-quality, context-aware results. Modify the generated rag_prompt.txt file if necessary, and run the script using "python3 query_for_results.py".

## Prerequisites
Ensure you have the required Python packages installed via the command "pip install -r requirements.txt". For Llama 2 model use (the default model), please ensure your access through Meta via Hugging Face, and configure your environment accordingly.

## Palmetto Cluster Instructions
Follow these steps for running the framework on the Palmetto Cluster:

1) Initiate an interactive job using the command: "qsub -I -l select=1:ncpus=16:ngpus=1:mem=64gb:gpu_model=a100 -l walltime=01:00:00".
2) Add the CUDA module with: "module add cuda/12.1.1-gcc/9.5.0".
3) Set up your conda environment using: "module add anaconda3/2022.05-gcc/9.5.0".
