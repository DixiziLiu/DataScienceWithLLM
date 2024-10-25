# Data Science and Large Language Models (LLM) for Healthcare Analytics

This repository contains the work and methods developed during research projects aimed at leveraging data science techniques and large language models (LLMs) to explore healthcare incident reports and patient feedback. The projects focus on uncovering actionable insights to improve patient safety, operational efficiency, and service quality in healthcare institutions.

## Project 1: Explore Stories Beyond Healthcare Incident Reports
This project integrates Retrieval-Augmented Generation (RAG) techniques and LLMs to retrieve and analyze data from 500 narratives and 70 million healthcare incident reports. The primary goal is to enhance decision-making support for hospital managers by optimizing large language models for patient safety and operational management.

### Key Contributions:
1) Data Collection: Conducted extensive interviews with 100 hospital staff from various departments (e.g., Prisma Health and Children’s Hospital) to gather insights into daily healthcare practices and challenges.
2) RAG Framework: Developed and deployed a RAG framework using GitHub for effective data retrieval and optimization of LLMs.
3) Model Optimization: Optimized Llama-3-70B and GPT-4o models using PyTorch and Hugging Face, incorporating prompt engineering and fine-tuning techniques to adapt models for specific hospital needs.
4) Impact: Achieved an 82% reduction in patient wait times through the implementation of data-driven AI response strategies, delivering actionable insights for hospital administrators.

## Project 2: Patient Feedback Analysis Using Large Language Models
This project utilized LLMs such as Llama and Chat-GPT4-O to analyze 70,000 patient comments, extracting qualitative and quantitative insights to understand patient expectations and improve satisfaction in healthcare services.

### Key Contributions:
1) Sentiment Analysis: Implemented sentiment analysis to derive insights on patient attitudes and expectations.
2) SQL Integration: Used SQL to extract demographic and feedback data, building visualizations with Google Analytics for insights into patient behavior and healthcare outcomes.
3) Regression Analysis: Applied R to perform regression analysis on patient feedback data, uncovering key factors driving patient satisfaction and service quality.
4) Impact: Insights from this analysis led to a 52% improvement in patient satisfaction, influencing targeted interventions to enhance hospital service.

## Repository Contents
/Data Analytics and Exploration: Contains all the core scripts for data preprocessing, model training, and feedback analysis.
**get_embeddings.py**: Extracts vector embeddings from datasets (CSV/JSON) for use in the RAG framework.
**make_a_prompt.py**: Generates prompts from the retrieved data to be fed into the large language models.
**query_for_results.py**: Queries LLMs using the generated prompts and retrieves results based on the input data.
/data: Sample data and incident reports used in the analysis (anonymized for privacy reasons). Cannot disclose because of privacy information.
/notebooks: Jupyter notebooks for step-by-step walkthroughs of patient feedback analysis, sentiment analysis, and regression models.
/results: Contains output files, visualizations, and model performance results.

## Prerequisites
Ensure you have the necessary Python packages installed:

pip install -r requirements.txt

Access to Llama 2 or Chat-GPT4-O via Hugging Face may be required. Please follow the respective documentation for acquiring access tokens and setting up the environment.

## Getting Started
To get started with this repository:

1) Clone the repository: git clone https://github.com/yourusername/Data-Science-and-LLM.git
2) Navigate to the project directory and install the dependencies: cd Data-Science-and-LLM; pip install -r requirements.txt
3) Explore the example scripts in the /src folder or run the provided notebooks in the /notebooks folder to see the data analysis pipeline in action.

Acknowledgments
This research was conducted as part of the Healthcare Data Analytics and Exploration work at Clemson University. Special thanks to Prisma Health and Children’s Hospital for their participation and support.

