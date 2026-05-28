import os
# Rezolvarea conflictului de librării OpenMP (OMP: Error #15) pe Mac local pentru PyTorch + FAISS
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import faiss
import numpy as np
import torch
import pickle
from transformers import AutoTokenizer, AutoModel
from pathlib import Path

# Selectăm dispozitivul: MPS pentru Apple Silicon (Mac), CUDA pentru GPU, altfel CPU
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
        """
        Încarcă modelul și tokenizatorul Transformer selectat.
        Se face lazy-loading pentru a nu încetini pornirea inițială a Streamlit.
        """
        if self.model is None:
            # Descarcă/încarcă modelul specificat de la Hugging Face
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModel.from_pretrained(self.model_name, attn_implementation="eager")
            self.model.to(DEVICE)
            self.model.eval() # Punem modelul în modul de evaluare (fără dropout)

    def tokenize(self, text):
        """
        Tokenizează textul primit folosind tokenizerul CodeBERT și returnează
        atât jetoanele (tokens) cât și ID-urile lor de vocabular.
        """
        self.load_model()
        tokens = self.tokenizer.tokenize(text)
        ids = self.tokenizer.convert_tokens_to_ids(tokens)
        return tokens, ids

    def get_attention_matrix(self, text):
        """
        Extrage matricea de auto-atenție (Self-Attention) din ultimul strat al CodeBERT.
        Returnează jetoanele (tokens) și matricea 2D de atenție medie.
        """
        self.load_model()
        
        # Tokenizăm textul (limitat pentru o afișare clară în grafic)
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=25)
        tokens = self.tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
        
        # Mutăm datele pe dispozitiv
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
        
        with torch.no_grad():
            # Rulăm modelul solicitând matricea de atenție
            outputs = self.model(**inputs, output_attentions=True)
            
            # Verificăm dacă atenția este disponibilă în output (pentru unele configurări)
            if hasattr(outputs, 'attentions') and outputs.attentions is not None:
                # outputs.attentions este un tuplu de 12 tensori de formă (batch_size, num_heads, seq_len, seq_len)
                # Extragem ultimul strat al Transformerului și primul element din batch
                last_layer_attention = outputs.attentions[-1][0]
                # Calculăm media ponderată pe toate cele 12 capete de atenție
                avg_attention = torch.mean(last_layer_attention, dim=0)
                return tokens, avg_attention.cpu().numpy()
            else:
                # Fallback în caz că nu e disponibilă
                seq_len = len(tokens)
                return tokens, np.eye(seq_len)

    def _mean_pooling(self, model_output, attention_mask):
        """
        Efectuează Mean Pooling pe hidden states ale Transformer-ului,
        luând în considerare masca de atenție (attention_mask).
        """
        token_embeddings = model_output[0] # Primul element conține toate stările ascunse ale tokenilor (hidden states)
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        return sum_embeddings / sum_mask

    def get_embeddings(self, texts, batch_size=8):
        """
        Generează embeddings vectoriale pentru o listă de texte folosind CodeBERT și PyTorch.
        """
        self.load_model()
        all_embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            
            # Tokenizare text cu trunchiere la 512 tokeni (limita standard a arhitecturii BERT)
            encoded_input = self.tokenizer(
                batch_texts, 
                padding=True, 
                truncation=True, 
                max_length=512, 
                return_tensors='pt'
            )
            
            # Mutăm datele pe GPU/MPS dacă este disponibil
            encoded_input = {k: v.to(DEVICE) for k, v in encoded_input.items()}
            
            with torch.no_grad():
                # Rulăm inferența pe arhitectura Transformer
                model_output = self.model(**encoded_input)
                
                # Extragem embedding-urile finale prin Mean Pooling peste tokeni
                batch_embeddings = self._mean_pooling(model_output, encoded_input['attention_mask'])
                
                # Copiem în memoria RAM ca numpy arrays
                all_embeddings.append(batch_embeddings.cpu().numpy())
                
        return np.vstack(all_embeddings)

    def _tokenize_text(self, text):
        import re
        return re.findall(r'[a-zA-Z_][a-zA-Z0-9_]*', text.lower())

    def build_lexical_index(self):
        """
        Construiește un index TF-IDF local și simplu pentru căutare lexicală rapidă,
        fără dependențe externe (zero dependency).
        """
        import math
        self.tf_idf_vectors = []
        self.idf = {}
        
        if not self.chunks:
            return
            
        num_documents = len(self.chunks)
        document_frequencies = {}
        
        # 1. Calculăm Frecvența Documentelor (DF)
        for idx, chunk in enumerate(self.chunks):
            content = chunk.get("content", "")
            # Combinăm cu numele elementului pentru a da greutate sporită numelui
            title_text = f"{chunk.get('name', '')} " * 5
            full_text = title_text + content
            tokens = set(self._tokenize_text(full_text))
            
            for token in tokens:
                document_frequencies[token] = document_frequencies.get(token, 0) + 1
                
        # 2. Calculăm IDF pentru fiecare cuvânt
        for token, df in document_frequencies.items():
            self.idf[token] = math.log((1 + num_documents) / (1 + df)) + 1
            
        # 3. Calculăm vectorii TF-IDF pentru fiecare document
        for chunk in self.chunks:
            content = chunk.get("content", "")
            title_text = f"{chunk.get('name', '')} " * 5
            full_text = title_text + content
            tokens = self._tokenize_text(full_text)
            
            # Frecvența Termenilor (TF)
            tf = {}
            for token in tokens:
                tf[token] = tf.get(token, 0) + 1
                
            # Vector TF-IDF
            tf_idf = {}
            for token, freq in tf.items():
                tf_norm = 1 + math.log(freq)
                tf_idf[token] = tf_norm * self.idf[token]
                
            # Calculăm norma L2 a vectorului pentru normalizare ulterioară
            l2_norm = math.sqrt(sum(val ** 2 for val in tf_idf.values())) + 1e-9
            normalized_tf_idf = {token: val / l2_norm for token, val in tf_idf.items()}
            
            self.tf_idf_vectors.append(normalized_tf_idf)

    def get_lexical_scores(self, query):
        """
        Calculează scorurile de similaritate lexicală (dot product de vectori TF-IDF)
        pentru toate chunk-urile în raport cu query-ul.
        """
        import math
        from pathlib import Path
        scores = np.zeros(len(self.chunks))
        if not self.chunks or not hasattr(self, 'tf_idf_vectors') or not self.tf_idf_vectors:
            return scores
            
        query_tokens = self._tokenize_text(query)
        if not query_tokens:
            return scores
            
        # Construim vectorul TF-IDF pentru query
        query_tf = {}
        for token in query_tokens:
            query_tf[token] = query_tf.get(token, 0) + 1
            
        query_tf_idf = {}
        for token, freq in query_tf.items():
            if token in self.idf:
                tf_norm = 1 + math.log(freq)
                query_tf_idf[token] = tf_norm * self.idf[token]
                
        # Normalizare L2 a vectorului query
        q_norm = math.sqrt(sum(val ** 2 for val in query_tf_idf.values())) + 1e-9
        normalized_query_tf_idf = {token: val / q_norm for token, val in query_tf_idf.items()}
        
        # Calculăm produsul scalar (dot product) cu fiecare document
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
            
            # 1. Exact or Substring name match boost (in both directions!)
            if q_lower and chunk_name:
                if q_lower == chunk_name:
                    dot_product += 2.0  # Absolute guarantee for exact match!
                elif q_lower in chunk_name or chunk_name in q_lower:
                    dot_product += 1.2  # Strong substring name match boost!
                    
            # 2. File path match boost
            if q_lower and (q_lower in file_name or q_lower in file_path):
                dot_product += 1.0  # File match boost!
                
            # 3. Code content literal substring match boost
            if q_lower and q_lower in content:
                dot_product += 0.8  # Strong literal code substring boost!

            # Cap individual lexical score at 1.0 to preserve elegant scale
            scores[idx] = min(1.0, dot_product)
            
        return scores

    def build_index(self, chunks, progress_bar_callback=None):
        """
        Primește o listă de chunk-uri, le generează embeddings și le adaugă în FAISS.
        """
        self.chunks = chunks
        if not chunks:
            return
            
        texts = [c["content"] for c in chunks]
        
        # Generăm vectorii folosind Transformer-ul
        embeddings = []
        batch_size = 4 # Batch size mic pentru a reduce consumul de RAM pe Mac local
        total_batches = (len(texts) + batch_size - 1) // batch_size
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            batch_emb = self.get_embeddings(batch_texts)
            embeddings.append(batch_emb)
            
            if progress_bar_callback:
                progress = min(1.0, (i + batch_size) / len(texts))
                progress_bar_callback(progress, f"Vectorizare blocuri cod... {i + len(batch_texts)} / {len(texts)}")
                
        all_embeddings = np.vstack(embeddings).astype('float32')
        
        # Dimensiunea modelului CodeBERT este de 768 dimensiuni
        dimension = all_embeddings.shape[1]
        
        # Creăm indexul FAISS L2 (L2 distance)
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(all_embeddings)
        
        # Construim indexul lexical local
        self.build_lexical_index()

    def save_index(self, save_dir):
        """
        Salvează indexul FAISS și metadatele chunks pe disc.
        """
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        
        # Salvare index FAISS
        if self.index is not None:
            faiss.write_index(self.index, str(save_path / "faiss_index.bin"))
            
        # Salvare chunk-uri (metadate și cod brut)
        with open(save_path / "chunks.pkl", "wb") as f:
            pickle.dump(self.chunks, f)

    def load_index(self, save_dir):
        """
        Încarcă indexul FAISS și chunk-urile de pe disc.
        """
        save_path = Path(save_dir)
        faiss_file = save_path / "faiss_index.bin"
        chunks_file = save_path / "chunks.pkl"
        
        if faiss_file.exists() and chunks_file.exists():
            self.index = faiss.read_index(str(faiss_file))
            with open(chunks_file, "rb") as f:
                self.chunks = pickle.load(f)
            # Reconstruim indexul lexical la încărcare
            self.build_lexical_index()
            return True
        return False

    def search(self, query, top_k=4, semantic_weight=0.5):
        """
        Efectuează o căutare hibridă (Lexicală TF-IDF + Semantică CodeBERT):
        1. Căutare semantică în indexul FAISS
        2. Căutare lexicală cu vectori TF-IDF localizați
        3. Combinare prin scor hibrid ponderat (prin semantic_weight)
        """
        if self.index is None or not self.chunks:
            return []
            
        # 1. Căutare Semantică (FAISS)
        query_vector = self.get_embeddings([query]).astype('float32')
        # Căutăm în FAISS pe toate documentele disponibile ca să le combinăm
        distances, indices = self.index.search(query_vector, len(self.chunks))
        
        semantic_scores = np.zeros(len(self.chunks))
        for i, idx in enumerate(indices[0]):
            if idx != -1 and idx < len(self.chunks):
                d = distances[0][i]
                # Convertim distanța L2 în scor de similaritate (0 la 1)
                semantic_scores[idx] = 1.0 / (1.0 + d)
                
        # 2. Căutare Lexicală (TF-IDF)
        if not hasattr(self, 'tf_idf_vectors') or not self.tf_idf_vectors:
            self.build_lexical_index()
            
        lexical_scores = self.get_lexical_scores(query)
        
        # 3. Combinare Ponderată Hibridă
        hybrid_results = []
        for idx in range(len(self.chunks)):
            s_sem = semantic_scores[idx]
            s_lex = lexical_scores[idx]
            # Combinăm cu ponderea dorită
            hybrid_score = semantic_weight * s_sem + (1.0 - semantic_weight) * s_lex
            
            chunk = self.chunks[idx].copy()
            chunk["score"] = float(hybrid_score)
            chunk["semantic_score"] = float(s_sem)
            chunk["lexical_score"] = float(s_lex)
            hybrid_results.append(chunk)
            
        # Sortăm descrescător după scorul hibrid final
        hybrid_results.sort(key=lambda x: x["score"], reverse=True)
        return hybrid_results[:top_k]
