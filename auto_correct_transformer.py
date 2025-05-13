import streamlit as st
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import torch

st.title("Correctify AI: Grammar Correction Tool")

# Load model and tokenizer
@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained("prithivida/grammar_error_correcter_v1")
    model = AutoModelForSeq2SeqLM.from_pretrained("prithivida/grammar_error_correcter_v1")
    return tokenizer, model

tokenizer, model = load_model()

# UI
user_input = st.text_area("Enter a sentence with errors:", "")

if st.button("Correct Sentence"):
    if user_input:
        input_ids = tokenizer.encode("gec: " + user_input, return_tensors="pt")
        outputs = model.generate(input_ids, max_length=128, num_beams=5, early_stopping=True)
        corrected_sentence = tokenizer.decode(outputs[0], skip_special_tokens=True)
        st.success("Corrected Sentence:")
        st.write(corrected_sentence)
    else:
        st.warning("Please enter a sentence.")