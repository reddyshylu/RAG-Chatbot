from pdf_extractor import extract_pdf_text
from chunker import chunk_text
from embedder import create_embeddings
from retriever import HybridRetriever
from generator import generate_reply
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer
import numpy as np

if __name__ == "__main__":
    full_text = extract_pdf_text('Chess For Dummies PDF.pdf')
    chunks = chunk_text(full_text, chunk_size=6)
    embeddings = create_embeddings(chunks)
    model = SentenceTransformer('all-MiniLM-L6-v2', device='cuda')
    tokenizer = AutoTokenizer.from_pretrained("gpt2-medium")

    # Initialize hybrid retriever (semantic + BM25)
    retriever = HybridRetriever(embeddings, chunks)

    EXAMPLES = """
    User: How does the bishop move?
    Assistant: The bishop moves diagonally across any number of squares. Each bishop is limited to squares of one color.

    User: How does the queen move?
    Assistant: The queen is the most powerful piece. She moves any number of squares horizontally, vertically, or diagonally.

    User: How do you checkmate with a queen and king?
    Assistant: To checkmate with a queen and king, use the queen to box in the enemy king, forcing him to the edge, then use your king to cover escape squares and deliver mate.
    ---
    """

    print("Welcome to Chess Studies RAG Bot! (type 'quit' to exit)")
    while True:
        query = input("\nYou: ")
        if query.lower().strip() == 'quit':
            break

        # Hybrid retrieval: k chunks each by vector and BM25, merged and deduplicated
        hybrid_chunks = retriever.retrieve_hybrid(query, model=model, k_bm25=3, k_vector=3)
        context = "\n".join(hybrid_chunks)

        prompt = (
            EXAMPLES +
            f"User: {query}\n"
            "Assistant: Based on the following context, provide a concise and clear answer to the user's question. "
            "If relevant, summarize key points and add examples. Context:\n"
            f"{context}\n"
            "Assistant:"
        )
        prompt_tokens = tokenizer.encode(prompt)
        max_prompt_tokens = 1024 - 180
        if len(prompt_tokens) > max_prompt_tokens:
            prompt_tokens = prompt_tokens[:max_prompt_tokens]
            prompt = tokenizer.decode(prompt_tokens)

        try:
            reply = generate_reply(prompt)
            print(reply)
        except Exception as e:
            print("Generator error:", e)
