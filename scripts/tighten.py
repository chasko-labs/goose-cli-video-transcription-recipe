#!/usr/bin/env python3
"""
tighten.py — remove filler language from transcripts and tighten narrative prose
Usage:
  tighten.py transcript <whisper.json> [output.json]   — regex filler removal
  tighten.py narrative <narrative.md> [ollama_url] [model]  — ollama prose tightening
"""
import sys
import os
import re
import json
import urllib.request
import urllib.error

# filler patterns — conservative, only strip unambiguous fillers
FILLER_PATTERNS = [
    # standalone fillers
    r'\b[Uu]h+\b',
    r'\b[Uu]m+\b',
    r'\b[Ee]rm\b',
    r'\b[Hh]mm+\b',
    r'\b[Aa]h+\b',
    # filler phrases
    r'\b[Yy]ou know,?\s*',
    r'\b[Ii] mean,?\s*',
    r'\b[Ss]o,?\s+(?=[a-z])',  # sentence-initial "so" before lowercase
    r'\b[Rr]ight\?\s*',
    r'\b[Yy]eah,?\s*',
    r'\b[Ll]ike,\s*',  # "like," with comma (filler usage)
]

# repeated word pattern
REPEATED_WORD = re.compile(r'\b(\w+)\s+\1\b', re.IGNORECASE)

# cleanup patterns for post-filler artifacts
CLEANUP_PATTERNS = [
    (re.compile(r'\s{2,}'), ' '),           # collapse multiple spaces
    (re.compile(r'\s+([,.:;!?])'), r'\1'),   # remove space before punctuation
    (re.compile(r',\s*,'), ','),             # collapse double commas
    (re.compile(r'^\s*[,;]\s*'), ''),        # strip leading punctuation after filler removal
    (re.compile(r'^\s+', re.MULTILINE), ''), # strip leading whitespace per line
]


def tighten_text(text):
    """remove filler words and clean up artifacts"""
    if not text or not text.strip():
        return text

    result = text
    for pattern in FILLER_PATTERNS:
        result = re.sub(pattern, '', result)

    # collapse repeated words ("the the" -> "the")
    result = REPEATED_WORD.sub(r'\1', result)

    for pattern, replacement in CLEANUP_PATTERNS:
        result = pattern.sub(replacement, result)

    return result.strip()


def tighten_transcript(input_path, output_path=None):
    """clean filler from whisper json — segments + full text"""
    data = json.loads(open(input_path).read())

    original_len = len(data.get("text", ""))
    data["text"] = tighten_text(data.get("text", ""))

    for seg in data.get("segments", []):
        seg["text"] = tighten_text(seg.get("text", ""))

    tightened_len = len(data.get("text", ""))
    removed = original_len - tightened_len
    pct = (removed / original_len * 100) if original_len > 0 else 0

    if output_path is None:
        # default: write alongside input as <base>-tightened.json
        base = input_path.rsplit('.', 1)[0]
        output_path = f"{base}-tightened.json"

    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print(f"[tighten] transcript: removed {removed} chars ({pct:.1f}%) -> {output_path}")
    return output_path


def tighten_narrative(md_path, ollama_url="http://localhost:11434", model="llama3.1:8b"):
    """send narrative through ollama for prose tightening"""
    text = open(md_path).read()

    # split header from body (header is everything before ---)
    parts = text.split("---\n", 1)
    if len(parts) == 2:
        header = parts[0] + "---\n"
        body = parts[1].strip()
    else:
        header = ""
        body = text.strip()

    if not body:
        print("[tighten] narrative: empty body, skipping")
        return

    prompt = f"""Tighten this narrative. Your ONLY job:
- remove filler language and hedging ("basically", "essentially", "it's worth noting")
- remove formulaic transitions ("The presenter then...", "He goes on to...", "Moving on...")
- remove redundant phrases and padding
- collapse wordy constructions into direct statements
- preserve ALL factual content, specific details, names, timestamps
- do NOT add new content or commentary
- do NOT change the structure or paragraph breaks
- return ONLY the tightened text, no preamble

Text to tighten:
{body}"""

    payload = json.dumps({
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0.1, "num_predict": 1200},
    }).encode()

    req = urllib.request.Request(
        f"{ollama_url}/api/generate",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            result = json.loads(resp.read())
            tightened = result.get("response", "").strip()
    except urllib.error.URLError as e:
        print(f"[tighten] narrative: ollama unreachable at {ollama_url}: {e}")
        return

    if not tightened:
        print("[tighten] narrative: empty response from ollama, skipping")
        return

    original_len = len(body)
    new_len = len(tightened)
    removed = original_len - new_len
    pct = (removed / original_len * 100) if original_len > 0 else 0

    with open(md_path, 'w') as f:
        f.write(header + "\n" + tightened + "\n")

    print(f"[tighten] narrative: {removed:+d} chars ({pct:+.1f}%) -> {md_path}")


def main():
    if len(sys.argv) < 3:
        print("usage: tighten.py transcript <whisper.json> [output.json]")
        print("       tighten.py narrative <narrative.md> [ollama_url] [model]")
        sys.exit(1)

    cmd = sys.argv[1]
    if cmd == "transcript":
        input_path = sys.argv[2]
        output_path = sys.argv[3] if len(sys.argv) > 3 else None
        tighten_transcript(input_path, output_path)
    elif cmd == "narrative":
        md_path = sys.argv[2]
        ollama_url = sys.argv[3] if len(sys.argv) > 3 else os.environ.get("OLLAMA_URL", "http://localhost:11434")
        model = sys.argv[4] if len(sys.argv) > 4 else os.environ.get("NARRATIVE_MODEL", "llama3.1:8b")
        tighten_narrative(md_path, ollama_url, model)


if __name__ == "__main__":
    main()
