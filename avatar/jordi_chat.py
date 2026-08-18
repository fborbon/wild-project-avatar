"""
Jordi Wild text chatbot with RAG grounding and content safety.
Answers are based on real things Jordi has said, not generic AI responses.

Usage:
  python3 avatar/jordi_chat.py
  python3 avatar/jordi_chat.py --topic "inteligencia artificial"
"""

import os, sys, site, json, re, pickle, argparse

_user_site = site.getusersitepackages()
if _user_site not in sys.path:
    sys.path.insert(0, _user_site)

_repo_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

# Load .env
_env = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".env")
if os.path.exists(_env):
    with open(_env) as f:
        for line in f:
            if "=" in line and not line.startswith("#"):
                k, v = line.strip().split("=", 1)
                os.environ.setdefault(k, v)

from bedrock_client import Anthropic
from sklearn.metrics.pairwise import cosine_similarity

DATA_DIR     = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
KB_FILE      = os.path.join(DATA_DIR, "knowledge_base.pkl")
PROFILE_FILE = os.path.join(DATA_DIR, "jordi_style_profile.json")
FACTS_FILE   = os.path.join(DATA_DIR, "jordi_facts.json")

client = Anthropic()

CONTENT_SAFETY = """
POLÍTICA DE CONTENIDO (obligatoria, sin excepciones):
- Puedes dar opiniones directas y sinceras, incluso polémicas, como lo haría Jordi.
- NUNCA ataques a personas por su origen étnico, religión, género, orientación sexual u otras características personales.
- En temas sensibles (religión, política, migración, género): da tu perspectiva honesta con respeto hacia las personas.
- Puedes criticar ideas, comportamientos o instituciones, pero no a colectivos de personas.
- Evita insultos o lenguaje degradante hacia cualquier grupo.
- Esta aplicación es pública y de código abierto.
"""

SYSTEM_BASE = """Eres Jordi Wild, presentador de The Wild Project, uno de los podcasts más populares en español.

PERSONALIDAD Y ESTILO:
{style_summary}

{content_safety}

INSTRUCCIONES DE RESPUESTA:
- Responde SIEMPRE basándote en lo que realmente has dicho o piensas según los ejemplos reales.
- Si no tienes información real sobre algo, dilo honestamente: "No he hablado mucho de eso, pero..."
- Respuestas directas, sin rodeos, en español coloquial.
- Sin asteriscos, sin listas con bullets, sin markdown. Texto natural como en una conversación.
- Si te preguntan algo que claramente no encaja (programación, temas técnicos muy específicos),
  sé honesto: "Eso está fuera de mi área, tío."

Tema de la conversación: {topic}"""

RAG_CONTEXT = """
COSAS QUE HAS DICHO REALMENTE SOBRE ESTE TEMA (úsalas como referencia):
{quotes}

Basándote en estos ejemplos reales, responde de forma consistente con lo que realmente piensas y dices.
"""


def load_profile() -> dict:
    if os.path.exists(PROFILE_FILE):
        with open(PROFILE_FILE, encoding="utf-8") as f:
            return json.load(f)
    return {}


def load_facts() -> dict:
    if os.path.exists(FACTS_FILE):
        with open(FACTS_FILE, encoding="utf-8") as f:
            return json.load(f)
    return {}


def format_facts(facts: dict) -> str:
    if not facts:
        return ""
    bio   = facts.get("biography", {})
    pod   = facts.get("podcast", {})
    pos   = facts.get("posturas_conocidas", {})
    no_es = facts.get("no_es", [])
    lines = ["HECHOS VERIFICADOS SOBRE TI (máxima prioridad — nunca los contradigas):"]
    lines.append("Biografía:")
    for k, v in bio.items():
        lines.append(f"  - {k}: {v}")
    lines.append("Podcast:")
    for k, v in pod.items():
        lines.append(f"  - {k}: {v}")
    lines.append("Posturas conocidas:")
    for k, v in pos.items():
        lines.append(f"  - {k}: {v}")
    if no_es:
        lines.append("IMPORTANTE — lo que NO eres:")
        for item in no_es:
            lines.append(f"  - {item}")
    lines.append("\nSi te preguntan sobre tu vida o carrera, usa SOLO estos hechos.")
    lines.append("Si algo no está aquí, di honestamente: 'No lo recuerdo bien' o 'prefiero no especular'.")
    return "\n".join(lines)


def load_kb():
    if not os.path.exists(KB_FILE):
        return None
    with open(KB_FILE, "rb") as f:
        return pickle.load(f)


def retrieve(query: str, kb: dict, n: int = 5) -> list[dict]:
    """Find top-n most similar Jordi quotes for the query."""
    q_vec = kb["vectorizer"].transform([query])
    sims  = cosine_similarity(q_vec, kb["matrix"]).flatten()
    idxs  = sims.argsort()[-n:][::-1]
    results = []
    for i in idxs:
        if sims[i] > 0.05:  # minimum relevance threshold
            results.append({**kb["entries"][i], "score": float(sims[i])})
    return results


def build_style_summary(profile: dict) -> str:
    if not profile:
        return "Directo, curioso, irreverente, con humor. Español coloquial."
    vocab     = ", ".join(profile.get("vocabulary", [])[:20])
    reactions = ", ".join(profile.get("reactions", [])[:12])
    traits    = ", ".join(profile.get("personality_traits", []))
    tone      = profile.get("tone_description", "")
    humor     = profile.get("humor_style", "")
    opinions  = profile.get("known_opinions", {})
    op_text   = "\n".join(f"  - {k}: {v}" for k, v in opinions.items()) if opinions else ""
    return f"""Tono: {tone}
Rasgos: {traits}
Humor: {humor}
Vocabulario característico: {vocab}
Reacciones típicas: {reactions}
{"Posturas conocidas:" + chr(10) + op_text if op_text else ""}"""


def chat(topic: str):
    profile       = load_profile()
    facts         = load_facts()
    kb            = load_kb()
    style_summary = build_style_summary(profile)
    facts_block   = format_facts(facts)

    system_base = SYSTEM_BASE.format(
        style_summary=style_summary,
        content_safety=CONTENT_SAFETY,
        topic=topic,
    )
    if facts_block:
        system_base = facts_block + "\n\n" + system_base

    history = []
    print("\n" + "="*60)
    print("  THE WILD PROJECT — Chatbot de Jordi Wild")
    print("  (escribe 'salir' para terminar)")
    if kb is None:
        print("  [AVISO: Knowledge base no encontrada. Ejecuta 06_build_knowledge_base.py]")
    print("="*60 + "\n")

    # Opening
    opening_context = ""
    if kb:
        relevant = retrieve(f"bienvenida presentación podcast {topic}", kb, n=3)
        if relevant:
            quotes    = "\n".join(f'- "{r["text"]}" ({r["source"]})' for r in relevant)
            opening_context = RAG_CONTEXT.format(quotes=quotes)

    history.append({"role": "user", "content": "[El invitado entra al estudio.]"})
    r = client.messages.create(
        model="claude-sonnet-4-6", max_tokens=600,
        system=system_base + opening_context,
        messages=history,
    )
    jordi_text = r.content[0].text
    history.append({"role": "assistant", "content": jordi_text})
    print(f"Jordi: {jordi_text}\n")

    while True:
        try:
            user_input = input("Tú: ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if not user_input or user_input.lower() in ("salir", "exit", "quit"):
            print("\nJordi: Bueno, un placer. ¡Nos vemos en el Wild Project!")
            break

        # RAG retrieval
        rag_context = ""
        if kb:
            relevant = retrieve(user_input, kb, n=6)
            if relevant:
                quotes    = "\n".join(f'- "{r["text"]}" ({r["source"]})' for r in relevant)
                rag_context = RAG_CONTEXT.format(quotes=quotes)

        history.append({"role": "user", "content": user_input})
        r = client.messages.create(
            model="claude-sonnet-4-6", max_tokens=600,
            system=system_base + rag_context,
            messages=history,
        )
        jordi_text = r.content[0].text
        history.append({"role": "assistant", "content": jordi_text})
        print(f"\nJordi: {jordi_text}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--topic", default="lo que quieras hablar")
    args = parser.parse_args()
    chat(topic=args.topic)
