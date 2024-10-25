# This contains the list of models and temperatures to be used for gathering augmented RETIPS data.
#Alpaca 30B models have an error with the bins and the vicuna models ignore the prompt
model_list = [\
              'huggyllama/llama-7b', 
              'huggyllama/llama-30b', 
              'Aeala/VicUnlocked-alpaca-30b',
              '/scratch/cehrett/hf_cache/models--alpaca-7b'
              # 'huggyllama/llama-13b', 
              # 'chavinlo/alpaca-native', 
              # 'anon8231489123/gpt4-x-alpaca-13b-native-4bit-128g', 
              # '4bit/gpt4-x-alpaca-13b-native-4bit-128g-cuda',  # Uncomment if above Alpaca 13b doesn't work
              # 'TheBloke/gpt4-alpaca-lora-30B-GPTQ-4bit-128g', 
              # 'MetaIX/GPT4-X-Alpaca-30B-4bit', 
              # 'nomic-ai/gpt4all-13b-snoozy', 
              # 'mosaicml/mpt-7b-instruct', 
              # 'mosaicml/mpt-7b',
             ]

temp_list = [\
             0.5,
             0.7,
             0.9,
             1.1,
             1.3,
             1.5,
            ]

aug_count_list = [\
                  10,
                  50,
                  100,
                  250,
                  500,
                 ]