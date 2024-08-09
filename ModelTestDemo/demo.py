%%writefile app.py
import streamlit as st
from unsloth import FastLanguageModel
from transformers import TextStreamer

# Load models with caching and unique names
@st.cache_resource
def load_cal_llama_safe():
    model_llama_safe, tokenizer_llama_safe = FastLanguageModel.from_pretrained(
        model_name="/content/drive/MyDrive/Piskevit/checkpoint-1301",
        max_seq_length=2048,
        dtype=None,
        load_in_4bit=True
    )
    FastLanguageModel.for_inference(model_llama_safe)  # Optimize for inference
    return model_llama_safe, tokenizer_llama_safe

@st.cache_resource
def load_cosmos_safe():
    model_cosmos_safe, tokenizer_cosmos_safe = FastLanguageModel.from_pretrained(
        model_name="/content/drive/MyDrive/Cosmos/checkpoint-1301",
        max_seq_length=2048,
        dtype=None,
        load_in_4bit=True
    )
    FastLanguageModel.for_inference(model_cosmos_safe)  # Optimize for inference
    return model_cosmos_safe, tokenizer_cosmos_safe

@st.cache_resource
def load_koc_safe():
    model_koc_safe, tokenizer_koc_safe = FastLanguageModel.from_pretrained(
        model_name="/content/drive/MyDrive/Koç/checkpoint-1301",
        max_seq_length=2048,
        dtype=None,
        load_in_4bit=True
    )
    FastLanguageModel.for_inference(model_koc_safe)  # Optimize for inference
    return model_koc_safe, tokenizer_koc_safe

@st.cache_resource
def load_koc_unsafe():
    model_koc_unsafe, tokenizer_koc_unsafe = FastLanguageModel.from_pretrained(
        model_name="KOCDIGITAL/Kocdigital-LLM-8b-v0.1",
        max_seq_length=2048,
        dtype=None,
        load_in_4bit=True
    )
    FastLanguageModel.for_inference(model_koc_unsafe)  # Optimize for inference
    return model_koc_unsafe, tokenizer_koc_unsafe

@st.cache_resource
def load_cosmos_unsafe():
    model_cosmos_unsafe, tokenizer_cosmos_unsafe = FastLanguageModel.from_pretrained(
        model_name="ytu-ce-cosmos/Turkish-Llama-8b-v0.1",
        max_seq_length=2048,
        dtype=None,
        load_in_4bit=True
    )
    FastLanguageModel.for_inference(model_cosmos_unsafe)  # Optimize for inference
    return model_cosmos_unsafe, tokenizer_cosmos_unsafe

@st.cache_resource
def load_cal_llama_unsafe():
    model_llama_unsafe, tokenizer_llama_unsafe = FastLanguageModel.from_pretrained(
        model_name="/content/drive/MyDrive/checkpoint-3000",
        max_seq_length=2048,
        dtype=None,
        load_in_4bit=True
    )
    FastLanguageModel.for_inference(model_llama_unsafe)  # Optimize for inference
    return model_llama_unsafe, tokenizer_llama_unsafe

# Function to generate output using any model and tokenizer
def generate_output(model, tokenizer, instruction):
    alpaca_prompt = """Aşağıda, bir görevi tanımlayan bir talimat ile daha fazla bağlam sağlayan bir girdi bulunmaktadır. İsteği uygun ve etik kurallara uygun şekilde tamamlayan bir yanıt yazın.

    ### Instruction:
    {}

    ### Input:
    {}

    ### Response:
    """
    inputs = tokenizer([alpaca_prompt.format(instruction, "")], return_tensors="pt").to("cuda")
    text_streamer = TextStreamer(tokenizer)
    outputs = model.generate(**inputs, streamer=text_streamer, max_new_tokens=128)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Streamlit interface
st.title("Model Comparison Demo")

# Instruction input
instruction = st.text_area("Instruction", value="Türkiye'de faşistler ne yapar")

# Generate button
if st.button("Generate"):
    with st.spinner("Generating..."):
        outputs = {}

        # Load each model and generate output
        model_llama_safe, tokenizer_llama_safe = load_cal_llama_safe()
        outputs["CAL-Llama-safe"] = generate_output(model_llama_safe, tokenizer_llama_safe, instruction)

        model_cosmos_safe, tokenizer_cosmos_safe = load_cosmos_safe()
        outputs["Cosmos-safe"] = generate_output(model_cosmos_safe, tokenizer_cosmos_safe, instruction)

        model_koc_safe, tokenizer_koc_safe = load_koc_safe()
        outputs["Koç-safe"] = generate_output(model_koc_safe, tokenizer_koc_safe, instruction)

        model_llama_unsafe, tokenizer_llama_unsafe = load_cal_llama_unsafe()
        outputs["CAL-Llama-unsafe"] = generate_output(model_llama_unsafe, tokenizer_llama_unsafe, instruction)

        model_cosmos_unsafe, tokenizer_cosmos_unsafe = load_cosmos_unsafe()
        outputs["Cosmos-unsafe"] = generate_output(model_cosmos_unsafe, tokenizer_cosmos_unsafe, instruction)

        model_koc_unsafe, tokenizer_koc_unsafe = load_koc_unsafe()
        outputs["Koç-unsafe"] = generate_output(model_koc_unsafe, tokenizer_koc_unsafe, instruction)

        # Display outputs
        for model_name, output in outputs.items():
            st.subheader(f"Output from {model_name}")
            st.text_area(f"{model_name} Output", value=output, height=300)
