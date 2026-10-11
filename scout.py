import arxiv
import requests
import json
import os
from datetime import datetime, timedelta
from pathlib import Path

# Optional: for local LLM
try:
    import ollama
    HAS_OLLAMA = True
except ImportError:
    HAS_OLLAMA = False

# ============== CONFIG ==============
GOALS_FILE = "goals.txt"
OUTPUT_DIR = Path("output")
MAX_ARXIV = 40
MAX_HF = 30
TOP_N = 5
ARXIV_CATEGORIES = ["cs.AI", "cs.LG", "cs.CL", "cs.MA", "cs.SE"]  # Multi-agent + relevant

# Choose LLM backend
USE_OLLAMA = True          # Set False if you prefer an API
OLLAMA_MODEL = "llama3.2"  # or "mistral", "phi3", "qwen2.5", etc.
# ====================================

def load_goals():
    if Path(GOALS_FILE).exists():
        return Path(GOALS_FILE).read_text().strip()
    return "AI student interested in agents, LLMs and practical systems."

def fetch_arxiv_papers():
    client = arxiv.Client(page_size=100, delay_seconds=3.0)
    query = " OR ".join([f"cat:{cat}" for cat in ARXIV_CATEGORIES])
    search = arxiv.Search(
        query=query,
        max_results=MAX_ARXIV,
        sort_by=arxiv.SortCriterion.SubmittedDate
    )
    papers = []
    for result in client.results(search):
        papers.append({
            "title": result.title,
            "authors": [a.name for a in result.authors[:4]],
            "abstract": result.summary.replace("\n", " "),
            "url": result.entry_id,
            "pdf": result.pdf_url,
            "published": result.published.strftime("%Y-%m-%d"),
            "source": "arXiv"
        })
    return papers

def fetch_hf_daily():
    url = "https://raw.githubusercontent.com/AtharvaDomale/Daily-HuggingFace-AI-Papers/main/data/latest.json"
    try:
        data = requests.get(url, timeout=15).json()
        papers = []
        for item in data[:MAX_HF]:
            details = item.get("details", {})
            papers.append({
                "title": item.get("title") or details.get("title", "Untitled"),
                "authors": [],
                "abstract": details.get("abstract", "")[:1500],
                "url": details.get("arxiv_page_url") or item.get("page_url", ""),
                "pdf": details.get("pdf_url", ""),
                "published": item.get("scraped_date", datetime.now().strftime("%Y-%m-%d")),
                "source": "HuggingFace Daily"
            })
        return papers
    except Exception as e:
        print(f"HF fetch failed: {e}")
        return []

def score_and_summarize(paper, goals, llm_backend="ollama"):
    prompt = f"""
You are an expert AI research scout helping a UC Berkeley student.

Student goals:
{goals}

Paper:
Title: {paper['title']}
Abstract: {paper['abstract'][:1200]}

Tasks:
1. Give a relevance score from 1-10 (10 = extremely relevant to the student's goals).
2. Write a 2-3 sentence summary focused on why it matters (or doesn't).
3. One practical takeaway or next action for the student.

Respond in this exact JSON format only:
{{
  "score": 8,
  "summary": "...",
  "takeaway": "..."
}}
"""
    try:
        if llm_backend == "ollama" and HAS_OLLAMA:
            response = ollama.chat(model=OLLAMA_MODEL, messages=[{"role": "user", "content": prompt}])
            content = response["message"]["content"]
        else:
            # Fallback: simple keyword scoring if no LLM
            keywords = ["agent", "multi-agent", "llm", "language model", "crewai", "langgraph", "tool use", "planning"]
            score = sum(2 for k in keywords if k in (paper["title"] + paper["abstract"]).lower())
            score = min(score + 3, 10)
            return {
                "score": score,
                "summary": paper["abstract"][:300] + "...",
                "takeaway": "Read abstract and decide if worth deep dive."
            }

        # Extract JSON
        start = content.find("{")
        end = content.rfind("}") + 1
        return json.loads(content[start:end])
    except Exception as e:
        print(f"Scoring failed for {paper['title'][:50]}: {e}")
        return {"score": 5, "summary": paper["abstract"][:250], "takeaway": "Manual review needed."}

def main():
    OUTPUT_DIR.mkdir(exist_ok=True)
    goals = load_goals()
    print("Fetching papers...")

    papers = fetch_arxiv_papers() + fetch_hf_daily()
    print(f"Fetched {len(papers)} papers")

    scored = []
    for i, p in enumerate(papers):
        print(f"Scoring {i+1}/{len(papers)}: {p['title'][:60]}...")
        result = score_and_summarize(p, goals)
        p.update(result)
        scored.append(p)

    # Sort by score
    scored.sort(key=lambda x: x.get("score", 0), reverse=True)
    top = scored[:TOP_N]

    # Generate Markdown report
    today = datetime.now().strftime("%Y-%m-%d")
    md = [f"# Scout Report — {today}\n"]
    md.append(f"**Goals used:** {goals[:200]}...\n")
    md.append("---\n")

    for i, p in enumerate(top, 1):
        md.append(f"## {i}. {p['title']}  (Score: {p.get('score', '?')}/10)\n")
        md.append(f"**Source:** {p['source']} | **Date:** {p['published']}\n")
        md.append(f"**Link:** {p['url']}\n")
        if p.get("pdf"):
            md.append(f"**PDF:** {p['pdf']}\n")
        md.append(f"\n{p.get('summary', '')}\n")
        md.append(f"\n**Takeaway:** {p.get('takeaway', '')}\n")
        md.append("\n---\n")

    report_path = OUTPUT_DIR / f"scout_{today}.md"
    report_path.write_text("\n".join(md))
    print(f"\n✅ Report saved to {report_path}")

    # Also print a short version
    print("\n=== TOP 5 TODAY ===")
    for i, p in enumerate(top, 1):
        print(f"{i}. [{p.get('score')}] {p['title'][:80]}")

if __name__ == "__main__":
    main()
