import arxiv
import requests
import json
import os
from datetime import datetime
from pathlib import Path

# ============== CONFIG ==============
GOALS_FILE = "goals.txt"
OUTPUT_DIR = Path("output")
MAX_ARXIV = 35
MAX_HF = 20
TOP_N = 5
ARXIV_CATEGORIES = ["cs.AI", "cs.LG", "cs.CL", "cs.MA"]
TO_EMAIL = "theojzap@berkeley.edu"
# ====================================

def load_goals():
    if Path(GOALS_FILE).exists():
        return Path(GOALS_FILE).read_text().strip()
    return "UC Berkeley student focused on AI agents, multi-agent systems, LLMs, and practical projects."

def fetch_arxiv_papers():
    client = arxiv.Client(page_size=50, delay_seconds=4.0)
    query = " OR ".join([f"cat:{cat}" for cat in ARXIV_CATEGORIES])
    search = arxiv.Search(
        query=query,
        max_results=MAX_ARXIV,
        sort_by=arxiv.SortCriterion.SubmittedDate
    )
    papers = []
    try:
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
    except Exception as e:
        print(f"arXiv warning: {e}")
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
                "abstract": details.get("abstract", "")[:1400],
                "url": details.get("arxiv_page_url") or item.get("page_url", ""),
                "pdf": details.get("pdf_url", ""),
                "published": item.get("scraped_date", datetime.now().strftime("%Y-%m-%d")),
                "source": "HuggingFace Daily"
            })
        return papers
    except Exception as e:
        print(f"HuggingFace fetch failed: {e}")
        return []

def score_paper(paper, goals):
    text = (paper["title"] + " " + paper["abstract"]).lower()
    
    high = ["agent", "multi-agent", "multi agent", "llm agent", "tool use", "tool-use",
            "planning", "langgraph", "crewai", "autogen", "agentic", "self-improvement",
            "recursive", "memory", "retrieval", "rag", "evaluation of agents"]
    medium = ["language model", "llm", "large language", "reasoning", "tool calling",
              "orchestration", "framework", "benchmark", "open-source"]
    
    score = 4
    for word in high:
        if word in text:
            score += 1.5
    for word in medium:
        if word in text:
            score += 0.7
            
    score = min(round(score), 10)
    
    abstract = paper["abstract"]
    summary = abstract[:320].rsplit(" ", 1)[0] + "..." if len(abstract) > 320 else abstract
    
    if score >= 8:
        takeaway = "High priority – strongly related to agents / multi-agent systems. Read the abstract and decide if you want to deep-dive or save for a project idea."
    elif score >= 6:
        takeaway = "Worth a quick look. Skim the abstract and PDF if the topic matches your current focus."
    else:
        takeaway = "Lower priority for now. You can skip unless the title strongly interests you."
    
    return {
        "score": score,
        "summary": summary,
        "takeaway": takeaway
    }

def send_email(report_md: str, today: str):
    """Send the report using Resend"""
    api_key = os.getenv("RESEND_API_KEY")
    if not api_key:
        print("No RESEND_API_KEY found – skipping email.")
        return

    try:
        import resend
        resend.api_key = api_key

        params = {
            "from": "Berkeley AI Scout <onboarding@resend.dev>",
            "to": [TO_EMAIL],
            "subject": f"Scout Report — {today}",
            "html": f"<pre style='font-family: monospace; white-space: pre-wrap;'>{report_md}</pre>"
        }

        email = resend.Emails.send(params)
        print(f"✅ Email sent successfully to {TO_EMAIL}")
    except Exception as e:
        print(f"❌ Email failed: {e}")

def main():
    OUTPUT_DIR.mkdir(exist_ok=True)
    goals = load_goals()
    print("Fetching papers...")

    papers = fetch_arxiv_papers() + fetch_hf_daily()
    print(f"Fetched {len(papers)} papers")

    # Deduplicate
    seen = set()
    unique = []
    for p in papers:
        t = p["title"].lower().strip()
        if t not in seen:
            seen.add(t)
            unique.append(p)

    scored = []
    for i, p in enumerate(unique):
        print(f"Scoring {i+1}/{len(unique)}: {p['title'][:60]}...")
        result = score_paper(p, goals)
        p.update(result)
        scored.append(p)

    scored.sort(key=lambda x: x["score"], reverse=True)
    top = scored[:TOP_N]
    today = datetime.now().strftime("%Y-%m-%d")

    # ===== Markdown Report =====
    md = [f"# Scout Report — {today}\n"]
    md.append(f"**For:** UC Berkeley AI student  \n**Papers reviewed:** {len(unique)}\n")
    md.append("---\n")

    for i, p in enumerate(top, 1):
        md.append(f"## {i}. {p['title']}\n")
        md.append(f"**Score:** {p['score']}/10  •  **Source:** {p['source']}  •  **Date:** {p['published']}\n")
        md.append(f"**Link:** {p['url']}\n")
        if p.get("pdf"):
            md.append(f"**PDF:** {p['pdf']}\n")
        md.append(f"\n### Summary\n{p['summary']}\n")
        md.append(f"\n### Recommended Action\n{p['takeaway']}\n")
        md.append("\n---\n")

    report_content = "\n".join(md)
    md_path = OUTPUT_DIR / f"scout_{today}.md"
    md_path.write_text(report_content)
    print(f"\n✅ Markdown saved → {md_path}")

    # ===== JSON =====
    json_data = {
        "date": today,
        "top_papers": [
            {
                "rank": i,
                "title": p["title"],
                "score": p["score"],
                "summary": p["summary"],
                "takeaway": p["takeaway"],
                "url": p["url"],
                "pdf": p.get("pdf", ""),
                "source": p["source"],
                "published": p["published"]
            }
            for i, p in enumerate(top, 1)
        ]
    }
    json_path = OUTPUT_DIR / f"scout_{today}.json"
    json_path.write_text(json.dumps(json_data, indent=2))
    print(f"✅ JSON saved     → {json_path}")

    # ===== Send Email =====
    send_email(report_content, today)

    print("\n=== TOP 5 TODAY ===")
    for i, p in enumerate(top, 1):
        print(f"{i}. [{p['score']}] {p['title'][:80]}")

if __name__ == "__main__":
    main()