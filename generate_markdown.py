import feedparser
from bot import generate_sw_comment  # your updated bot.py
from datetime import date
import subprocess

# === CONFIG ===
RSS_URL = 'https://news.google.com/rss/search?q="AI+lawyer"+OR+"legal+AI"+OR+"lawtech"+OR+"legaltech"+OR+"AI+regulation"+OR+"legal+automation"+OR+"AI+compliance"'
MAX_POSTS = 2

# === Fetch news ===
feed = feedparser.parse(RSS_URL)
entries = feed.entries[:MAX_POSTS]  # limit to 2 posts

if not entries:
    print("No news items found today.")
    exit(0)

# === Generate markdown content ===
today = date.today().isoformat()
file_name = f"news-{today}.md"

content_lines = [f"# Swetlana Daily Comments — {today}\n"]

for entry in entries:
    title = entry.title
    summary = getattr(entry, "summary", "")  # some feeds may have summary
    url = entry.link
    comment = generate_sw_comment(title, summary, url)

    content_lines.append(f"## {title}\n")
    content_lines.append(f"**Link:** {url}\n")
    content_lines.append(f"**Summary:** {summary}\n")
    content_lines.append(f"**Swetlana Commentary:**\n{comment}\n")
    content_lines.append("---\n")

content = "\n".join(content_lines)

# === Save markdown file ===
with open(file_name, "w", encoding="utf-8") as f:
    f.write(content)

print(f"Daily file generated: {file_name}")

# === Commit & push to GitHub ===
subprocess.run(["git", "config", "user.name", "github-actions[bot]"], check=True)
subprocess.run(["git", "config", "user.email", "github-actions[bot]@users.noreply.github.com"], check=True)
subprocess.run(["git", "add", file_name], check=True)
subprocess.run(["git", "commit", "-m", f"Daily Swetlana Comments update: {today}"], check=True)
subprocess.run(["git", "push", "origin", "main"], check=True)
