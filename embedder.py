from sentence_transformers import SentenceTransformer

def create_embeddings(chunks, model_name='all-MiniLM-L6-v2', device='cuda'):
    model = SentenceTransformer(model_name, device=device)
    embeddings = model.encode(chunks, batch_size=32, show_progress_bar=True)
    return embeddings
