"""
Analyzes transcripts to extract Jordi Wild's style profile.
Since we don't have diarization yet, this uses Claude to:
  1. Identify which lines are likely Jordi's (the interviewer).
  2. Extract characteristic phrases, questions, reactions, and patterns.
Output: ./data/jordi_style_profile.json  and  ./data/jordi_lines.txt

Requires: AWS credentials with bedrock:InvokeModel (see bedrock_client.py)
"""

import os
import re
import json
import glob
import random
from bedrock_client import Anthropic
from dotenv import load_dotenv

load_dotenv()

TRANSCRIPTS_DIR = "./transcripts"
OUT_DIR = "./data"
SAMPLE_SIZE = 5          # episodes to analyze (increase once confirmed working)
CHUNK_CHARS = 6000       # characters per API call

os.makedirs(OUT_DIR, exist_ok=True)
client = Anthropic()


DIARIZE_PROMPT = """\
Este es un fragmento de transcripción de un podcast de entrevistas en español.
El entrevistador es Jordi Wild. Hay un invitado diferente en cada episodio.

Identifica y extrae SOLO las intervenciones de Jordi Wild (el entrevistador).
Devuelve una lista JSON de strings, cada uno siendo una frase o turno de Jordi.
No incluyas nada del invitado.

Transcripción:
{chunk}

Devuelve SOLO el JSON, sin explicación. Ejemplo: ["frase 1", "frase 2"]
"""

STYLE_PROMPT = """\
Eres un analista de estilo conversacional. Analiza estas intervenciones de Jordi Wild,
presentador del podcast The Wild Project, y extrae un perfil detallado de su estilo.

Intervenciones de Jordi:
{jordi_lines}

Devuelve un JSON con esta estructura exacta:
{{
  "vocabulary": ["palabras o expresiones características"],
  "opening_moves": ["formas típicas de iniciar preguntas o temas"],
  "reactions": ["expresiones de sorpresa, acuerdo, desacuerdo, humor"],
  "challenge_patterns": ["cómo cuestiona o debate con el invitado"],
  "humor_style": "descripción del tipo de humor que usa",
  "topic_transitions": ["frases para cambiar de tema"],
  "personality_traits": ["rasgos de personalidad detectables en el habla"],
  "sample_questions": ["10 preguntas representativas tal como las formularía Jordi"],
  "tone_description": "descripción general del tono y registro"
}}

Devuelve SOLO el JSON.
"""


def chunk_text(text: str, size: int) -> list[str]:
    return [text[i:i+size] for i in range(0, len(text), size)]


def extract_jordi_lines(transcript: str) -> list[str]:
    lines = []
    for chunk in chunk_text(transcript, CHUNK_CHARS):
        response = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=2048,
            messages=[{"role": "user", "content": DIARIZE_PROMPT.format(chunk=chunk)}],
        )
        raw = response.content[0].text.strip()
        try:
            raw = re.sub(r"^```(?:json)?\s*", "", raw)
            raw = re.sub(r"\s*```$", "", raw)
            parsed = json.loads(raw)
            if isinstance(parsed, list):
                lines.extend(parsed)
        except json.JSONDecodeError:
            pass
    return lines


def build_style_profile(all_jordi_lines: list[str]) -> dict:
    sample = random.sample(all_jordi_lines, min(200, len(all_jordi_lines)))
    joined = "\n".join(f"- {l}" for l in sample)
    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=4096,
        messages=[{"role": "user", "content": STYLE_PROMPT.format(jordi_lines=joined)}],
    )
    raw = response.content[0].text.strip()
    # strip markdown code fences if present
    raw = re.sub(r"^```(?:json)?\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)
    return json.loads(raw)


def main():
    txt_files = sorted(
        glob.glob(os.path.join(TRANSCRIPTS_DIR, "*.txt")) +
        glob.glob(os.path.join(TRANSCRIPTS_DIR, "**", "*.txt"))
    )
    if not txt_files:
        print(f"No transcripts found in {TRANSCRIPTS_DIR}/. Run 02_parse_transcripts.py first.")
        return

    sample_files = random.sample(txt_files, min(SAMPLE_SIZE, len(txt_files)))
    print(f"Analyzing {len(sample_files)} episodes for Jordi's lines...")

    all_jordi_lines = []
    for path in sample_files:
        name = os.path.basename(path)
        print(f"  Processing: {name[:70]}")
        with open(path, encoding="utf-8") as f:
            text = f.read()
        lines = extract_jordi_lines(text)
        all_jordi_lines.extend(lines)
        print(f"    -> {len(lines)} Jordi turns extracted")

    # Save raw lines
    jordi_lines_path = os.path.join(OUT_DIR, "jordi_lines.txt")
    with open(jordi_lines_path, "w", encoding="utf-8") as f:
        f.write("\n".join(all_jordi_lines))
    print(f"\nSaved {len(all_jordi_lines)} Jordi lines to {jordi_lines_path}")

    # Build style profile
    print("Building style profile...")
    profile = build_style_profile(all_jordi_lines)
    profile_path = os.path.join(OUT_DIR, "jordi_style_profile.json")
    with open(profile_path, "w", encoding="utf-8") as f:
        json.dump(profile, f, ensure_ascii=False, indent=2)
    print(f"Style profile saved to {profile_path}")


if __name__ == "__main__":
    main()
