import streamlit as st
from transformers import pipeline
from transformers import AutoTokenizer, AutoModel
import pandas as pd
import torch
from transformers import EncoderDecoderModel
from tqdm import tqdm
from datasets import Dataset,load_dataset
tqdm.pandas()
from dataclasses import dataclass, field
from transformers import TrainingArguments
from typing import Optional
import datasets
from transformers import Seq2SeqTrainer

model_path='Safaya_BERTMid_BERTMid_BBC'

@st.cache(allow_output_mutation=True)
def load_summarizer():
    model_name = "asafaya/bert-medium-arabic"
    model = EncoderDecoderModel.from_encoder_decoder_pretrained(model_name, model_name, tie_encoder_decoder=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # set special tokens
    model.config.decoder_start_token_id = tokenizer.cls_token_id
    model.config.eos_token_id = tokenizer.sep_token_id
    model.config.pad_token_id = tokenizer.pad_token_id

    # sensible parameters for beam search
    model.config.vocab_size = model.config.decoder.vocab_size
    model.config.max_length = 64
    # model.config.min_length = 56
    model.config.no_repeat_ngram_size = 3
    model.config.early_stopping = True
    model.config.length_penalty = 2.0
    model.config.num_beams = 4
    model.load_state_dict(torch.load("./"+ model_path + "/model"))
    model.eval()
    return model,tokenizer
def generate_summary(text):
    # Tokenizer will automatically set [BOS] <text> [EOS]
    # cut off at BERT max length 512
    processed_text=text #add preprocessing
    inputs = tokenizer(processed_text, padding="max_length", truncation=True, max_length=512, return_tensors="pt",add_special_tokens=True)
    input_ids = inputs.input_ids.to("cuda:0")
    attention_mask = inputs.attention_mask.to("cuda:0")
    model.to("cuda:0")
    outputs = model.generate(input_ids, attention_mask=attention_mask)

    # all special tokens including will be removed
    output_str = tokenizer.batch_decode(outputs, skip_special_tokens=True)



    return output_str

model,tokenizer = load_summarizer()
st.title("Summarize Text")
sentence = st.text_area('Please paste your article :', height=30)
button = st.button("Summarize")
#
# max = st.sidebar.slider('Select max', 50, 500, step=10, value=150)
# min = st.sidebar.slider('Select min', 10, 450, step=10, value=50)
# do_sample = st.sidebar.checkbox("Do sample", value=False)
with st.spinner("Generating Summary.."):
    if button and sentence:
        res = generate_summary(sentence)[0].replace('"','')

        # text = ' '.join([summ['summary_text'] for summ in res])
        # st.write(result[0]['summary_text'])
        st.write(res)