import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import faiss
import numpy as np
import torch
import pickle
from transformers import AutoTokenizer, AutoModel
from pathlib import Path


if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")

class CodeBERTIndexer:
    def __init__(self, model_name="microsoft/codebert-base"):
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self.index = None
        self.chunks = []

    def load_model(self):
\
\
\

        if self.model is None:

            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModel.from_pretrained(self.model_name, attn_implementation="eager")
            self.model.to(DEVICE)
            self.model.eval()

    def tokenize(self, text):
\
\
\

        self.load_model()
        tokens = self.tokenizer.tokenize(text)
        ids = self.tokenizer.convert_tokens_to_ids(tokens)
        return tokens, ids

    def get_attention_matrix(self, text):
\
\
\

        self.load_model()


        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=25)
        tokens = self.tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])


        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

        with torch.no_grad():

            outputs = self.model(**inputs, output_attentions=True)


            if hasattr(outputs, 'attentions') and outputs.attentions is not None:


                last_layer_attention = outputs.attentions[-1][0]

                avg_attention = torch.mean(last_layer_attention, dim=0)
                return tokens, avg_attention.cpu().numpy()
            else:

                seq_len = len(tokens)
                return tokens, np.eye(seq_len)

    def _mean_pooling(self, model_output, attention_mask):
\
\
\

        token_embeddings = model_output[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        return sum_embeddings / sum_mask

    def get_embeddings(self, texts, batch_size=8):
\
\

        self.load_model()
        all_embeddings = []

        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]


            encoded_input = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors='pt'
            )


            encoded_input = {k: v.to(DEVICE) for k, v in encoded_input.items()}

            with torch.no_grad():

                model_output = self.model(**encoded_input)


                batch_embeddings = self._mean_pooling(model_output, encoded_input['attention_mask'])


                all_embeddings.append(batch_embeddings.cpu().numpy())

        return np.vstack(all_embeddings)

    def _tokenize_text(self, text):
        import re
        return re.findall(r'[a-zA-Z_][a-zA-Z0-9_]*', text.lower())

    def build_lexical_index(self):
\
\
\

        import math
        self.tf_idf_vectors = []
        self.idf = {}

        if not self.chunks:
            return

        num_documents = len(self.chunks)
        document_frequencies = {}


        for idx, chunk in enumerate(self.chunks):
            content = chunk.get("content", "")

            title_text = f"{chunk.get('name', '')} " * 5
            full_text = title_text + content
            tokens = set(self._tokenize_text(full_text))

            for token in tokens:
                document_frequencies[token] = document_frequencies.get(token, 0) + 1


        for token, df in document_frequencies.items():
            self.idf[token] = math.log((1 + num_documents) / (1 + df)) + 1


        for chunk in self.chunks:
            content = chunk.get("content", "")
            title_text = f"{chunk.get('name', '')} " * 5
            full_text = title_text + content
            tokens = self._tokenize_text(full_text)


            tf = {}
            for token in tokens:
                tf[token] = tf.get(token, 0) + 1


            tf_idf = {}
            for token, freq in tf.items():
                tf_norm = 1 + math.log(freq)
                tf_idf[token] = tf_norm * self.idf[token]


            l2_norm = math.sqrt(sum(val ** 2 for val in tf_idf.values())) + 1e-9
            normalized_tf_idf = {token: val / l2_norm for token, val in tf_idf.items()}

            self.tf_idf_vectors.append(normalized_tf_idf)

    def get_lexical_scores(self, query):
\
\
\

        import math
        from pathlib import Path
        scores = np.zeros(len(self.chunks))
        if not self.chunks or not hasattr(self, 'tf_idf_vectors') or not self.tf_idf_vectors:
            return scores

        query_tokens = self._tokenize_text(query)
        if not query_tokens:
            return scores


        query_tf = {}
        for token in query_tokens:
            query_tf[token] = query_tf.get(token, 0) + 1

        query_tf_idf = {}
        for token, freq in query_tf.items():
            if token in self.idf:
                tf_norm = 1 + math.log(freq)
                query_tf_idf[token] = tf_norm * self.idf[token]


        q_norm = math.sqrt(sum(val ** 2 for val in query_tf_idf.values())) + 1e-9
        normalized_query_tf_idf = {token: val / q_norm for token, val in query_tf_idf.items()}


        for idx, doc_vector in enumerate(self.tf_idf_vectors):
            dot_product = 0.0
            for token, q_val in normalized_query_tf_idf.items():
                if token in doc_vector:
                    dot_product += q_val * doc_vector[token]

            q_lower = query.lower().strip()
            chunk_name = self.chunks[idx].get("name", "").lower()
            content = self.chunks[idx].get("content", "").lower()
            file_path = self.chunks[idx].get("file_path", "").lower()
            file_name = Path(file_path).name.lower()


            if q_lower and chunk_name:
                if q_lower == chunk_name:
                    dot_product += 2.0
                elif q_lower in chunk_name or chunk_name in q_lower:
                    dot_product += 1.2


            if q_lower and (q_lower in file_name or q_lower in file_path):
                dot_product += 1.0


            if q_lower and q_lower in content:
                dot_product += 0.8


            scores[idx] = min(1.0, dot_product)

        return scores

    def build_index(self, chunks, progress_bar_callback=None):
\
\

        self.chunks = chunks
        if not chunks:
            return

        texts = [c["content"] for c in chunks]


        embeddings = []
        batch_size = 4
        total_batches = (len(texts) + batch_size - 1) // batch_size

        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            batch_emb = self.get_embeddings(batch_texts)
            embeddings.append(batch_emb)

            if progress_bar_callback:
                progress = min(1.0, (i + batch_size) / len(texts))
                progress_bar_callback(progress, f"Incarcare blocuri de cod... {i + len(batch_texts)} / {len(texts)}")

        all_embeddings = np.vstack(embeddings).astype('float32')


        dimension = all_embeddings.shape[1]


        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(all_embeddings)


        self.build_lexical_index()

    def save_index(self, save_dir):
\
\

        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)


        if self.index is not None:
            faiss.write_index(self.index, str(save_path / "faiss_index.bin"))


        with open(save_path / "chunks.pkl", "wb") as f:
            pickle.dump(self.chunks, f)

    def load_index(self, save_dir):
\
\

        save_path = Path(save_dir)
        faiss_file = save_path / "faiss_index.bin"
        chunks_file = save_path / "chunks.pkl"

        if faiss_file.exists() and chunks_file.exists():
            self.index = faiss.read_index(str(faiss_file))
            with open(chunks_file, "rb") as f:
                self.chunks = pickle.load(f)

            self.build_lexical_index()
            return True
        return False

    def search(self, query, top_k=4, semantic_weight=0.5):
\
\
\
\
\

        if self.index is None or not self.chunks:
            return []


        query_vector = self.get_embeddings([query]).astype('float32')

        distances, indices = self.index.search(query_vector, len(self.chunks))

        semantic_scores = np.zeros(len(self.chunks))
        for i, idx in enumerate(indices[0]):
            if idx != -1 and idx < len(self.chunks):
                d = distances[0][i]

                semantic_scores[idx] = 1.0 / (1.0 + d)


        if not hasattr(self, 'tf_idf_vectors') or not self.tf_idf_vectors:
            self.build_lexical_index()

        lexical_scores = self.get_lexical_scores(query)


        hybrid_results = []
        for idx in range(len(self.chunks)):
            s_sem = semantic_scores[idx]
            s_lex = lexical_scores[idx]

            hybrid_score = semantic_weight * s_sem + (1.0 - semantic_weight) * s_lex

            chunk = self.chunks[idx].copy()
            chunk["score"] = float(hybrid_score)
            chunk["semantic_score"] = float(s_sem)
            chunk["lexical_score"] = float(s_lex)
            hybrid_results.append(chunk)


        hybrid_results.sort(key=lambda x: x["score"], reverse=True)
        return hybrid_results[:top_k]
