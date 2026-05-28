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
            print('-----------------------' + str(self.model.config) + '-----------------------') #Afoisăm configurația modelului pentru debug

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
            outputs = self.model(**inputs, output_attentions=True, return_dict=True)
            
            # Verificăm dacă atenția este disponibilă în output (pentru unele configurări)
            #if hasattr(outputs, 'attentions') and outputs.attentions is not None and len(outputs.attentions) > 0:
            if hasattr(outputs, 'attentions') and outputs.attentions is not None:
                # outputs.attentions este un tuplu de 12 tensori de formă (batch_size, num_heads, seq_len, seq_len)
                # Extragem ultimul strat al Transformerului și primul element din batch
                last_layer_attention = outputs.attentions[-1][0]
                # Calculăm media ponderată pe toate cele 12 capete de atenție
                avg_attention = torch.mean(last_layer_attention, dim=0)
                print("-----------------TOKENS:-------------------")
                print(tokens)

                print("------------ATTENTION SHAPE:---------------")
                print(avg_attention.shape)

                print("------------ATTENTION MATRIX:---------------")
                print(avg_attention.cpu().numpy())

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
        faiss.normalize_L2(all_embeddings)

        dimension = all_embeddings.shape[1]
        
        # Creăm indexul FAISS bazat pe similaritate cosinus
        #self.index = faiss.IndexFlatL2(dimension)
        self.index = faiss.IndexFlatIP(dimension)
        self.index.add(all_embeddings)

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
            return True
        return False

    def search(self, query, top_k=4):
        """
        Efectuează o căutare semantică: 
        1. Tokenizează query-ul și obține vectorul prin CodeBERT
        2. Caută în indexul FAISS top_k cele mai apropiate bucăți de cod
        3. Returnează rezultatele cu metadate
        """
        if self.index is None or not self.chunks:
            return []
            
        # Generăm embedding-ul pentru întrebare
        query_vector = self.get_embeddings([query]).astype('float32')
        faiss.normalize_L2(query_vector)
        
        # Căutăm în FAISS
        similarities, indices = self.index.search(query_vector, top_k)
        
        results = []
        for i, idx in enumerate(indices[0]):
            if idx != -1 and idx < len(self.chunks):
                chunk = self.chunks[idx].copy()
                chunk["score"] = float(similarities[0][i])
                results.append(chunk)
                
        return results
