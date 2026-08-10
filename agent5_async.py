import re
import urllib.parse
from ddgs import DDGS
import asyncio
import aiohttp                  # ← NEW: Async HTTP client
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import time
import chromadb                 # ← NEW: Vector DB for persistence
from chromadb.config import Settings

# DEFAULT CONSTANTS (unchanged)
DEFAULT_SEARCH_RESULTS = 13
DEFAULT_PASSAGES_PER_PAGE = 4
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_TOP_PASSAGES = 5
DEFAULT_SUMMARY_SENTENCES = 3
DEFAULT_TIMEOUT = 16

def unwrap_ddg(url):  # unchanged
    try:
        parsed = urllib.parse.urlparse(url)
        if "duckduckgo.com" in parsed.netloc:
            qs = urllib.parse.parse_qs(parsed.query)
            uddg = qs.get("uddg")
            if uddg:
                return urllib.parse.unquote(uddg[0])
    except Exception:
        pass
    return url

def search_web(query, max_results=DEFAULT_SEARCH_RESULTS):  # unchanged
    urls = []
    with DDGS() as ddgs:
        for r in ddgs.text(query, max_results=max_results):
            url = r.get("href") or r.get("url")
            if url:
                urls.append(unwrap_ddg(url))
    return urls

# NEW ASYNC FETCH FUNCTION
async def fetch_text_async(session, url, timeout=DEFAULT_TIMEOUT):
    """Async version of fetch_text using aiohttp"""
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    }
    try:
        async with session.get(url, timeout=timeout, headers=headers, allow_redirects=True) as resp:
            print(f"Fetching: {url} → Status: {resp.status}")
            if resp.status != 200:
                return ""
            text = await resp.text()
            ct = resp.headers.get("content-type", "").lower()
            if not any(x in ct for x in ["html", "text"]):
                return ""
            
            soup = BeautifulSoup(text, "html.parser")
            for tag in soup(["script", "style", "noscript", "header", "footer", "svg", "iframe", "nav", "aside"]):
                tag.extract()
            
            paragraphs = [p.get_text(" ", strip=True) for p in soup.find_all("p")]
            content = " ".join(p for p in paragraphs if p)
            cleaned = re.sub(r"\s+", " ", content).strip()
            print(f"  → Success: {len(cleaned)} chars")
            return cleaned
    except Exception as e:
        print(f"  → Error {url}: {type(e).__name__}")
        return ""

class ShortResearchAgent:
    def __init__(self, embed_model=EMBEDDING_MODEL, persist_dir="./chroma_research_db"):
        print(f"Loading embedder: {embed_model}...")
        self.embedder = SentenceTransformer(embed_model)
        
        # NEW: ChromaDB Persistence
        self.persist_dir = persist_dir
        self.chroma_client = chromadb.PersistentClient(path=persist_dir)
        self.collection = self.chroma_client.get_or_create_collection("research_passages")
        print(f"ChromaDB initialized at {persist_dir}")

    # NEW: Async batch fetching
    async def _gather_pages(self, urls, timeout):
        async with aiohttp.ClientSession() as session:
            tasks = [fetch_text_async(session, u, timeout) for u in urls]
            return await asyncio.gather(*tasks)   # Parallel fetching
        
        
    # changed to async def run()
    async def run(self, query: str, search_results=DEFAULT_SEARCH_RESULTS, 
            passages_per_page=DEFAULT_PASSAGES_PER_PAGE,
            top_passages=DEFAULT_TOP_PASSAGES, 
            summary_sentences=DEFAULT_SUMMARY_SENTENCES, 
            timeout=DEFAULT_TIMEOUT):
        
            start = time.time()
            urls = search_web(query, max_results=search_results)
            print(f"Found {len(urls)} URLs.")

            # ASYNC FETCH - this is the main performance win
            # loop = asyncio.get_event_loop()
            # texts = loop.run_until_complete(self._gather_pages(urls, timeout))
            # --- FIXED: Use clean native await instead of extracting loop ---
            texts = await self._gather_pages(urls, timeout)

            # Rest of the logic is almost identical (chunking, embedding, ranking)
            docs = []
            for u, txt in zip(urls, texts):
                if not txt:
                    continue
                chunks = chunk_passages(txt, max_words=120, overlap=20)
                for c in chunks[:passages_per_page]:
                    docs.append({"url": u, "passage": c})

            if not docs:
                return {"query": query, "passages": [], "summary": "No content fetched."}

            # Embedding & ranking (unchanged core logic)
            texts = [d["passage"] for d in docs]
           # emb_txts = self.embedder.encode(texts, convert_to_numpy=True)
            #q_emb = self.embedder.encode(query, convert_to_numpy=True)
            
            # NEW ASYNC EMBEDDINGS
            emb_txts = await asyncio.to_thread(self.embedder.encode, texts, show_progress_bar=True, convert_to_numpy=True)
            q_emb = await asyncio.to_thread(self.embedder.encode, query, show_progress_bar=True, convert_to_numpy=True)

            def cosine_sklearn(a, b):
                a = np.asarray(a).reshape(1, -1)
                b = np.asarray(b).reshape(1, -1)
                return cosine_similarity(a, b)[0][0]

            sims = [cosine_sklearn(e, q_emb) for e in emb_txts]
            top_idx = np.argsort(sims)[::-1][:top_passages]
            top_passages_list = [{"url": docs[i]["url"], "passage": docs[i]["passage"], "score": float(sims[i])} for i in top_idx]

            # NEW: Persist top passages to ChromaDB
            for p in top_passages_list:
                self.collection.add(
                    documents=[p["passage"]],
                    metadatas=[{"url": p["url"], "query": query, "timestamp": time.time()}],
                    ids=[f"passage_{int(time.time()*1000)}"]
                )

            # Summary logic (unchanged)
            # ... (same sentence reranking as original)
            # 5. Rerank passages to passages to create a mini summary
            sentences = []
            for tp in top_passages_list:
                for s in split_sentences(tp["passage"]):
                    sentences.append({"sent": s, "url": tp["url"]})
                    
            if not sentences:
                summary = "No summary could be generated"
            else:
                sent_texts = [s["sent"]for s in sentences]
                #sent_embs = self.embedder.encode(sent_texts, convert_to_numpy=True, show_progress_bar=True)
                # NEW ASYNCHRONOUS NON BLOCKING Offload summary sentence matrix multiplication to an internal worker thread
                sent_embs = await asyncio.to_thread(self.embedder.encode, sent_texts, convert_to_numpy=True, show_progress_bar=True)
                sent_sims = [cosine_sklearn(e, q_emb) for e in sent_embs]
                
                # 6. Select top sentences for summary
                top_sent_idx = np.argsort(sent_sims)[::-1][:DEFAULT_SUMMARY_SENTENCES]
                chosen = [sentences[idx] for idx in top_sent_idx]
                
                # deduplicate and format
                seen = set()
                lines = []
                for s in chosen:
                    key = s["sent"].lower()[:80] # check first 80 chars
                    if key in seen:
                        continue # continue to next item (key) in the loop
                    seen.add(key)
                    lines.append(f"{s['sent']} (Source: {s['url']})")
                summary = " ".join(lines)

            elapsed = time.time() - start
            return {
                "query": query,
                "passages": top_passages_list,
                "summary": summary,
                "time": elapsed,
                "params_used": {
                    "search_results": search_results,
                    "passages_per_page": passages_per_page,
                    "top_passages": top_passages,
                    "summary_sentences": summary_sentences,
                    "timeout": timeout}   
            }

# Helper functions (moved down, unchanged)
def chunk_passages(text, max_words=120, overlap=20):
    """Split long text into smaller passages."""
    words = text.split()
    if not words:
        return []
    chunks = []
    i = 0
    while i < len(words):
        chunk = words[i: i + max_words]
        chunks.append(" ".join(chunk))
        i += max_words - overlap
        if i < 0:
            i = max_words - overlap // 2
    return chunks
    
def split_sentences(text):
    """A simple sentence splitter."""
    parts = re.split(r'(?<=[.!?])\s+', text)
    return [p.strip() for p in parts if p.strip()]


if __name__ == "__main__":
    async def main():
        agent = ShortResearchAgent()
        # Test async run
        result = await agent.run("What causes urban heat islands and how can cities reduce them?")
        print(result["summary"])
        
    asyncio.run(main())