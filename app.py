import streamlit as st
from typing import List, Tuple

from src.infer import load_pipeline, predict

@st.cache_resource
def get_pipeline():
    return load_pipeline()

def label_color(label: str) -> str:
    upper_label = label.upper()
    if "PROBLEM" in upper_label:
        return "#FF6B6B"  
    if "TREATMENT" in upper_label:
        return "#4ECDC4"  
    if "TEST" in upper_label:
        return "#45B7D1"  
    return "#95A5A6" 

def render_entities(entities: List[Tuple[str, str]]) -> str:
    parts = []
    for token, label in entities:
        if label == "O":
            parts.append(f"<span style='padding:2px 4px;'>{token}</span>")
        else:
            color = label_color(label)
            entity_type = label.split("-")[-1]
            parts.append(
                f"<span style='background-color:{color};"
                f"color:white;padding:2px 4px;border-radius:4px;margin:2px;"
                f"font-weight:500;'>{token} "
                f"<sub style='font-size:0.7em;opacity:0.9;'>{entity_type}</sub></span>"
            )
    return " ".join(parts)

def main():
    st.set_page_config(page_title="Medical NER", layout="wide")
    st.title("Medical NER")

    text_input = st.text_area("Enter medical text:", height=200)
    
    if st.button("Run NER") and text_input.strip():
        with st.spinner("Running"):
            model, tokenizer, label2id, id2label = get_pipeline()
            entities = predict(text_input, model, tokenizer, id2label)
            st.markdown(render_entities(entities), unsafe_allow_html=True)

if __name__ == "__main__":
    main()
