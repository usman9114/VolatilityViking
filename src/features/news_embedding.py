import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import torch
import os

class NewsEmbedder:
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        # Mac MPS support
        if torch.backends.mps.is_available():
            self.device = 'mps'
            
        print(f"Loading SentenceTransformer: {model_name} on {self.device}...")
        self.model = SentenceTransformer(model_name, device=self.device)

    def process_news(self, news_path, timeframe='4h'):
        """
        Loads news, encodes, and resamples to timeframe.
        """
        print(f"Loading news from {news_path}...")
        df = pd.read_parquet(news_path)
        
        # Ensure UTC
        if df.index.tz is None:
            df.index = df.index.tz_localize('UTC')
        else:
            df.index = df.index.tz_convert('UTC')
            
        # Encode headlines
        print("Encoding headlines (this may take a while)...")
        # Batch processing is automatic in sentence_transformers usually, but let's be safe
        headlines = df['headline'].tolist()
        
        # Encode
        embeddings = self.model.encode(headlines, show_progress_bar=True, batch_size=64)
        
        # Create DataFrame of embeddings
        emb_df = pd.DataFrame(embeddings, index=df.index)
        emb_df.columns = [f'emb_{i}' for i in range(emb_df.shape[1])]
        
        # Resample to align with candle data
        # We need to aggregating multiple news items in the same 4h block.
        # Mean pooling is standard.
        print(f"Resampling to {timeframe}...")
        resampled_df = emb_df.resample(timeframe).mean()
        
        # Forward fill empty periods (no news -> assume sentiment persists? Or 0?)
        # For embeddings, 0 vector is neutral-ish but technically 'no info'.
        # Let's fill with 0s for now or ffill. 
        # ffill makes sense for "prevailing sentiment".
        resampled_df.fillna(method='ffill', inplace=True)
        resampled_df.fillna(0, inplace=True) # For initial NaNs
        
        return resampled_df

    def save_embeddings(self, df, filepath):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        df.to_parquet(filepath)
        print(f"Embeddings saved to {filepath}")

if __name__ == "__main__":
    embedder = NewsEmbedder()
    df_emb = embedder.process_news('data/raw/crypto_news.parquet')
    print(df_emb.head())
    print(df_emb.shape)
    embedder.save_embeddings(df_emb, 'data/processed/news_embeddings.parquet')
