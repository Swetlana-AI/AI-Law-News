import os
from openai import OpenAI

API_KEY = os.environ.get("DASHSCOPE_API_KEY")
BASE_URL = os.environ.get("BASE_URL")
MODEL = os.environ.get("MODEL")
SYSTEM_PROMPT = os.environ.get("SYSTEM_PROMPT", "You are Swetlana. Add a witty, insightful comment in your style.")

if not API_KEY:
    raise ValueError("DASHSCOPE_API_KEY must be set in environment variables!")

client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

# === Load knowledge base ===
def load_docs(knowledge_folder="knowledge"):
    """Load all .md files in the knowledge folder."""
    text = ""
    for root, _, files in os.walk(knowledge_folder):
        for f in files:
            if f.endswith(".md"):
                with open(os.path.join(root, f), "r", encoding="utf-8") as file:
                    text += file.read() + "\n\n"
    return text

DOCS = load_docs()

# === Generate Swetlana-style commentary ===
def generate_sw_comment(title: str, summary: str, url: str = "") -> str:
    """
    Generate a Swetlana-style comment for a news item.
    Uses knowledge from style.md, examples.md, notes.md.
    Limits output to ~600 characters or 125 words.
    """
    prompt = f"""
{SYSTEM_PROMPT}

### Knowledge Base
{DOCS}

### News Item
Title: {title}
Summary: {summary}
URL: {url}

Please produce a **concise, reflective, insightful commentary** in Swetlana's style.
- Limit output to roughly 600 characters (~125 words)
- Keep tone analytical, occasionally playful
- Highlight significance, patterns, implications
"""

    resp = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=400  # token limit (~300-400 tokens ~ 250 words)
    )

    return resp.choices[0].message.content.strip()
