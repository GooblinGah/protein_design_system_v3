# apps/provenance_app.py
import json, streamlit as st
st.set_page_config(page_title="Provenance Visualizer", layout="wide")
st.title("Per-residue provenance visualizer")
log_path = st.text_input("Ledger path", "data_pipeline/data/safety_ledger_generated.jsonl")
entry_idx = st.number_input("Entry index (0-based)", 0, step=1)
if st.button("Load"):
    lines = open(log_path).read().strip().splitlines()
    rec = json.loads(lines[int(entry_idx)])
    seq = rec.get("sequence",""); prov = rec.get("provenance", []); exemplars = rec.get("exemplars", [])
    st.write(f"Exemplars: {exemplars}")
    colored = []
    for i,c in enumerate(seq):
        src = prov[i].get("source","vocab") if i < len(prov) else "vocab"
        color = "#ffd54f" if src=="copy" else "#e0e0e0"
        colored.append(f"<span title='{src}' style='background:{color}'>{c}</span>")
    st.markdown("<div style='font-family:monospace;font-size:16px'>"+"".join(colored)+"</div>", unsafe_allow_html=True)
