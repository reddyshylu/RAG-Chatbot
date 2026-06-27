from nltk.tokenize import sent_tokenize

def chunk_text(text, chunk_size=6):
    sentences = sent_tokenize(text)
    return [' '.join(sentences[i:i+chunk_size]) for i in range(0, len(sentences), chunk_size)]
