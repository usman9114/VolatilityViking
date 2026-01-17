import os
import logging
import chromadb
import pandas as pd
from typing import List, Dict

logger = logging.getLogger(__name__)

class MarketMemory:
    """
    The Global Pattern Store.
    Stores historical market snapshots as semantic vectors.
    """
    def __init__(self, db_path="data/chroma_db", collection_name="global_patterns"):
        self.client = chromadb.PersistentClient(path=db_path)
        
        # Use OpenAI Embeddings (Professional Grade)
        from chromadb.utils import embedding_functions
        openai_ef = embedding_functions.OpenAIEmbeddingFunction(
            api_key=os.getenv("OPENAI_API_KEY"),
            model_name="text-embedding-3-small"
        )
        
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            embedding_function=openai_ef,
            metadata={"hnsw:space": "cosine"}
        )
        logger.info(f"[MarketMemory] Connected to collection '{collection_name}' with {self.collection.count()} items.")

    def semanticize_snapshot(self, row: pd.Series, symbol: str) -> str:
        """
        Converts a row of numerical indicators into a 'Rich Text' market description.
        This is the Matryoshka-style embedding structure.
        """
        # 1. Market Regime (The "Container")
        trend = "Neutral"
        if row['ema_20'] > row['ema_50'] > row['ema_200']: trend = "Strong Uptrend"
        elif row['ema_20'] < row['ema_50'] < row['ema_200']: trend = "Strong Downtrend"
        
        # 2. Volatility
        volatility = "Normal"
        if row['bb_width'] > 0.3: volatility = "Explosive" # Arbitrary threshold, tune later
        elif row['bb_width'] < 0.05: volatility = "Squeeze"
        
        # 3. Momentum
        rsi_state = "Neutral"
        if row['rsi'] > 75: rsi_state = "Overbought"
        elif row['rsi'] < 25: rsi_state = "Oversold"
        
        # 4. Volume
        volume = "Average"
        # Assuming obv_slope is normalized, > 0.5 is high accumulation
        if row['obv_slope'] > 0.5: volume = "Accumulation"
        elif row['obv_slope'] < -0.5: volume = "Distribution"
        
        # Construct the detailed text for the LLM/Embedding model
        # OPTIMIZATION: Removed "Asset: {symbol}" to allow cross-asset pattern matching.
        # Now a DOGE chart can match a historical BTC pattern if the structure is identical.
        text = (
            f"[REGIME: {trend}] [VOLATILITY: {volatility}] [MOMENTUM: {rsi_state}] "
            # f"Asset: {symbol}. " # Removed for universal matching
            f"Details: RSI is {row['rsi']:.1f}. Price is {row['dist_ema_200']*100:.1f}% from 200 EMA. "
            f"ADX is {row.get('adx', 0):.1f}. " # ADX might be missing if simple script
            f"OBV shows {volume}. "
            f"Day of week: {int(row['day_of_week'])}."
        )
        return text

    def ingest_batch(self, df: pd.DataFrame, symbol: str):
        """
        Batch process a dataframe and store it in Chroma.
        """
        documents = []
        metadatas = []
        ids = []
        
        # We process every Nth candle to save space/cost, or all of them? 
        # For 3 years of 1h data (26k rows), we can do all. 
        # Total 15 coins * 26k = 390k.
        # But for embedding cost ($0.02/1M tokens), 390k * 100 tokens = 39M tokens = $0.80. Very cheap.
        
        logger.info(f"[MarketMemory] Ingesting {len(df)} snapshots for {symbol}...")
        
        batch_size = 500
        for i, (index, row) in enumerate(df.iterrows()):
            # Create Description
            text = self.semanticize_snapshot(row, symbol)
            
            # Create Metadata (The "Answer Key")
            meta = {
                "symbol": symbol,
                "timestamp": int(row['timestamp']),
                "ret_4h": float(row['future_ret_4h']),
                "ret_24h": float(row['future_ret_24h']),
                "rsi": float(row['rsi']),
                "regime": text.split(']')[0] # Hack to get regime quickly
            }
            
            # Create ID
            doc_id = f"{symbol}_{int(row['timestamp'])}"
            
            documents.append(text)
            metadatas.append(meta)
            ids.append(doc_id)
            
            if len(documents) >= batch_size:
                self.collection.upsert(
                    documents=documents,
                    metadatas=metadatas,
                    ids=ids
                )
                documents = []
                metadatas = []
                ids = []
                if i % 5000 == 0:
                    logger.info(f"  Indexed {i}/{len(df)}...")

        # Flush remaining
        if documents:
            self.collection.upsert(documents=documents, metadatas=metadatas, ids=ids)
        
        logger.info(f"[MarketMemory] Finished ingesting {symbol}.")

    def query_similar_patterns(self, current_snapshot_text: str, n_results=10) -> List[Dict]:
        """
        Finds historical scenarios that match the current one.
        Returns a list of Dicts with 'metadata' containing outcomes.
        """
        results = self.collection.query(
            query_texts=[current_snapshot_text],
            n_results=n_results
        )
        
        # Flatten results
        # results['metadatas'] is [[{},{},...]]
        hit_list = []
        if results['metadatas']:
            for meta, dist in zip(results['metadatas'][0], results['distances'][0]):
                hit_list.append({
                    "metadata": meta,
                    "distance": dist # Check Chroma distance metric (usually smaller = closer)
                })
                
        return hit_list
