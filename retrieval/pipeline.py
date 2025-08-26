# retrieval/pipeline.py
from .faiss_index import Retriever, kmer_embed
from .msa_builder import build_msa
from utils import load_fasta

def build_profile_from_db(db_fasta, query_seq, topk=16):
    recs = list(load_fasta(db_fasta))
    if not recs: return [], []
    names = [n for n,_ in recs]; seqs=[s for _,s in recs]
    R = Retriever(names, seqs)
    hits = R.query(query_seq, topk=topk)  # [(name, seq, sim)]
    msa = build_msa(hits)
    return msa, hits  # hits include (id, seq, similarity)
