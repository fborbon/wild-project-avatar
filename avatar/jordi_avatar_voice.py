"""
Jordi Wild animated voice avatar — lightweight version.
- Whisper tiny  → fast transcription (~2s on CPU)
- edge-tts      → fast Spanish TTS (~0.5s, needs internet)
- Amazon Nova (AWS Bedrock) → Jordi's brain
- pygame        → animated face window

Controls:
  SPACE  — hold to speak, release to send
  ESC/Q  — quit

Usage:
  python3 avatar/jordi_avatar_voice.py --topic "tu vida y carrera"
  python3 avatar/jordi_avatar_voice.py --topic "deporte" --photo jordi.jpeg
"""

import os, sys, site, json, time, re, tempfile, threading, argparse, asyncio

# Ensure user site-packages are visible even inside a virtual environment
_user_site = site.getusersitepackages()
if _user_site not in sys.path:
    sys.path.insert(0, _user_site)

import numpy as np
import pygame
import cv2
import sounddevice as sd
import scipy.io.wavfile as wavfile
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from face_animator import FaceAnimator

# ── Load .env ─────────────────────────────────────────────────────────────────
_env = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".env")
if os.path.exists(_env):
    with open(_env) as _f:
        for _l in _f:
            if "=" in _l and not _l.startswith("#"):
                k, v = _l.strip().split("=", 1)
                os.environ.setdefault(k, v)

_repo_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from bedrock_client import Anthropic
import edge_tts
from faster_whisper import WhisperModel

# ── Config ────────────────────────────────────────────────────────────────────
SAMPLE_RATE   = 16000
WIN_W, WIN_H  = 720, 580
FPS           = 30
FACE_SIZE     = 300   # animated face square in pixels
PROFILE_PATH  = "./data/jordi_style_profile.json"
TTS_VOICE     = "es-ES-AlvaroNeural"   # Spanish male voice
MIC_GAIN      = 0.1                    # reduce mic input (1.0 = full, lower if saturating)

BG      = (15, 15, 20)
ORANGE  = (255, 90, 20)
GREEN   = (0, 210, 100)
BLUE    = (80, 150, 255)
WHITE   = (240, 240, 240)
GREY    = (90, 90, 90)

IDLE, LISTENING, PROCESSING, SPEAKING = "idle","listening","processing","speaking"

# ── Helpers ───────────────────────────────────────────────────────────────────

def load_profile():
    if os.path.exists(PROFILE_PATH):
        with open(PROFILE_PATH, encoding="utf-8") as f:
            return json.load(f)
    return {}

def load_facts() -> str:
    facts_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "jordi_facts.json")
    if not os.path.exists(facts_path):
        return ""
    with open(facts_path, encoding="utf-8") as f:
        facts = json.load(f)
    bio   = facts.get("biography", {})
    no_es = facts.get("no_es", [])
    pos   = facts.get("posturas_conocidas", {})
    lines = ["HECHOS VERIFICADOS (máxima prioridad — nunca los contradigas):"]
    for k, v in bio.items():
        lines.append(f"  - {k}: {v}")
    if no_es:
        lines.append("Lo que NO eres: " + "; ".join(no_es))
    for k, v in pos.items():
        lines.append(f"  - {k}: {v}")
    lines.append("Para preguntas sobre tu vida usa SOLO estos hechos.")
    return "\n".join(lines)


def build_system_prompt(profile, topic):
    if not profile:
        return (
            "Eres Jordi Wild de The Wild Project. Entrevistas con estilo directo, "
            "curioso, irreverente y con humor. Español coloquial. "
            "Respuestas MUY CORTAS: 2-3 frases. Sin listas ni asteriscos."
        )
    vocab     = ", ".join(profile.get("vocabulary", [])[:15])
    reactions = ", ".join(profile.get("reactions", [])[:10])
    traits    = ", ".join(profile.get("personality_traits", []))
    samples   = "\n".join(f'- "{q}"' for q in profile.get("sample_questions", [])[:5])
    return f"""Eres Jordi Wild, presentador de The Wild Project.

RASGOS: {traits}
VOCABULARIO: {vocab}
REACCIONES: {reactions}
PREGUNTAS DE REFERENCIA:
{samples}

REGLAS:
- Respuestas MUY CORTAS: 2-3 frases. Esto se va a escuchar, no a leer.
- Una sola pregunta por turno. Sin asteriscos ni markdown.
- Reacciona antes de preguntar. Habla como Jordi, no como un asistente.
- Tema de hoy: {topic}"""

def wrap(text, font, max_w):
    words, lines, cur = text.split(), [], ""
    for w in words:
        t = (cur + " " + w).strip()
        if font.size(t)[0] <= max_w:
            cur = t
        else:
            if cur: lines.append(cur)
            cur = w
    if cur: lines.append(cur)
    return lines

_xtts_model = None   # loaded once, reused across all synthesis calls

def synth_tts(text: str, mode: str = "edge") -> str:
    """Synthesize text, return path to wav file. mode: 'edge' or 'xtts'."""
    global _xtts_model
    clean = re.sub(r"\*[^*]+\*", "", text)
    clean = re.sub(r"\s+", " ", clean).strip()

    if mode == "xtts":
        from TTS.api import TTS as CoquiTTS
        voice_sample = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "jordi_voice_sample.wav")
        if _xtts_model is None:
            _xtts_model = CoquiTTS("tts_models/multilingual/multi-dataset/xtts_v2", gpu=False)
        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        tmp.close()
        if os.path.exists(voice_sample):
            _xtts_model.tts_to_file(text=clean, speaker_wav=voice_sample, language="es", file_path=tmp.name)
        else:
            _xtts_model.tts_to_file(text=clean, file_path=tmp.name)
        return tmp.name

    # Default: edge-tts
    tmp = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False)
    tmp.close()

    async def _run():
        comm = edge_tts.Communicate(clean, TTS_VOICE)
        await comm.save(tmp.name)

    asyncio.run(_run())
    # Convert mp3 → wav for sounddevice
    wav_path = tmp.name.replace(".mp3", ".wav")
    os.system(f'ffmpeg -y -i "{tmp.name}" -ar 22050 -ac 1 "{wav_path}" -loglevel quiet')
    os.unlink(tmp.name)
    return wav_path

# ── Avatar ────────────────────────────────────────────────────────────────────

class JordiAvatar:
    def __init__(self, topic, photo_path):
        self.topic       = topic
        self.state       = IDLE
        self.subtitle    = ""
        self.ring_phase  = 0.0
        self.current_amp = 0.0      # shared between _speak thread and draw loop
        self.conversation    = []
        self.recording_buf   = []
        self.is_recording    = False
        self.loading_msg     = "Cargando Whisper tiny..."
        self.models_ready    = False
        self.face_animator   = None  # set after photo loaded
        self._last_frame_t   = time.time()

        self.tts_mode = "edge"   # overridden by main() from --tts arg
        self.claude   = Anthropic()
        facts = load_facts()
        self.system = (facts + "\n\n" if facts else "") + build_system_prompt(load_profile(), topic)
        self.whisper = None

        # Load face animator from photo if provided
        if photo_path and os.path.exists(photo_path):
            try:
                img = cv2.imread(photo_path)
                self.face_animator = FaceAnimator(img, size=FACE_SIZE)
                print("Face animator ready.")
            except Exception as e:
                print(f"Face animator failed: {e}. Falling back to circle avatar.")

        threading.Thread(target=self._load_models, daemon=True).start()

    def _load_models(self):
        self.loading_msg = "Cargando Whisper base..."
        self.whisper = WhisperModel("base", device="cpu", compute_type="int8")

        if self.tts_mode == "xtts":
            from TTS.utils.manage import ModelManager
            import os
            model_dir = os.path.join(os.path.expanduser("~"), ".local", "share", "tts",
                                     "tts_models--multilingual--multi-dataset--xtts_v2")
            if os.path.exists(os.path.join(model_dir, "model.pth")):
                self.loading_msg = "Cargando XTTS v2 desde caché (~15s)..."
            else:
                self.loading_msg = "Descargando XTTS v2 por primera vez (~1.8 GB)..."
            synth_tts(".", mode="xtts")   # warm up — loads model into _xtts_model

        self.loading_msg  = ""
        self.models_ready = True
        threading.Thread(target=self._jordi_open, daemon=True).start()

    def _jordi_open(self):
        self.state = PROCESSING
        self.conversation.append({"role":"user","content":"[El invitado entra al estudio.]"})
        try:
            r = self.claude.messages.create(
                model="claude-sonnet-4-6", max_tokens=200,
                system=self.system, messages=self.conversation)
            if not r.content:
                raise ValueError(f"Empty response (stop_reason={r.stop_reason})")
            text = r.content[0].text
        except Exception as e:
            self.subtitle = f"[Error API: {e}]"
            self.state = LISTENING
            return
        self.conversation.append({"role":"assistant","content":text})
        self._speak(text)

    def _speak(self, text):
        self.state    = SPEAKING
        self.subtitle = text
        try:
            wav_path = synth_tts(text, mode=self.tts_mode)
            sr, data = wavfile.read(wav_path)
            os.unlink(wav_path)
            if data.dtype != np.float32:
                data = data.astype(np.float32) / 32768.0
            self.current_amp = 1.0
            sd.play(data, samplerate=sr)
            chunk = sr // FPS
            idx = 0
            while idx < len(data):
                chunk_data = data[idx:idx+chunk]
                self.current_amp = min(1.0, max(0.0, float(np.abs(chunk_data).mean() * 15)))
                idx += chunk
                time.sleep(1 / FPS)
            sd.wait()
        except Exception as e:
            self.subtitle = f"[Error TTS: {e}]"
        self.current_amp = 0.0
        self.state       = LISTENING

    def start_recording(self):
        if self.state != LISTENING: return
        self.recording_buf = []
        self.is_recording  = True
        self.subtitle = "Escuchando..."

    def stop_recording(self):
        if not self.is_recording: return
        self.is_recording = False
        if len(self.recording_buf) < 2:
            self.state = LISTENING
            return
        audio = np.concatenate(self.recording_buf)
        threading.Thread(target=self._process, args=(audio,), daemon=True).start()

    def audio_callback(self, indata, frames, time_info, status):
        if self.is_recording:
            self.recording_buf.append((indata * MIC_GAIN).flatten())

    def _process(self, audio):
        self.state    = PROCESSING
        self.subtitle = "Transcribiendo..."
        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        tmp.close()
        # Normalize to consistent level for Whisper
        peak = np.abs(audio).max()
        if peak > 0.01:
            audio = audio / peak * 0.5
        wavfile.write(tmp.name, SAMPLE_RATE, (audio * 32767).astype(np.int16))
        try:
            segs, _ = self.whisper.transcribe(
                tmp.name,
                language="es",
                task="transcribe",
                beam_size=5,
                vad_filter=True,
            )
            user_text = " ".join(s.text for s in segs).strip()
        finally:
            os.unlink(tmp.name)
        if not user_text:
            self.state = LISTENING
            return
        self.subtitle = f"Tú: {user_text}"
        time.sleep(0.5)
        self.conversation.append({"role":"user","content":user_text})
        try:
            r = self.claude.messages.create(
                model="claude-sonnet-4-6", max_tokens=200,
                system=self.system, messages=self.conversation)
            if not r.content:
                raise ValueError(f"Empty response (stop_reason={r.stop_reason})")
            jordi = r.content[0].text
        except Exception as e:
            self.subtitle = f"[Error API: {e}]"
            self.conversation.pop()   # remove unanswered user turn
            self.state = LISTENING
            return
        self.conversation.append({"role":"assistant","content":jordi})
        self._speak(jordi)

    # ── Draw ──────────────────────────────────────────────────────────────────

    def draw(self, screen, fonts):
        screen.fill(BG)
        now = time.time()
        dt  = now - self._last_frame_t
        self._last_frame_t = now
        cx, cy = WIN_W // 2, WIN_H // 2 - 40

        # ── Animated face ─────────────────────────────────────────────────
        if self.face_animator:
            amp   = self.current_amp if self.state == SPEAKING else 0.0
            frame = self.face_animator.get_frame(amp, dt)
            surf  = FaceAnimator.bgr_to_pygame(frame)
            screen.blit(surf, surf.get_rect(center=(cx, cy)))
        else:
            pygame.draw.circle(screen, (40, 40, 50), (cx, cy), 130)
            lbl = fonts["big"].render("JORDI", True, GREY)
            screen.blit(lbl, lbl.get_rect(center=(cx, cy)))

        # ── Ring overlays (speaking / listening / processing) ─────────────
        face_r = FACE_SIZE // 2 + 8

        if self.state == SPEAKING:
            self.ring_phase += 0.1
            for i in range(3):
                ph = self.ring_phase - i * 0.7
                r  = face_r + int(20 * np.sin(ph) * max(0.2, self.current_amp)) + i * 14
                a  = max(0, int(140 * (1 - i * 0.3) * abs(np.sin(ph))))
                s  = pygame.Surface((WIN_W, WIN_H), pygame.SRCALPHA)
                pygame.draw.circle(s, (*ORANGE, a), (cx, cy), r, 3)
                screen.blit(s, (0, 0))

        elif self.state == LISTENING:
            pulse = 0.5 + 0.5 * np.sin(now * 5)
            s = pygame.Surface((WIN_W, WIN_H), pygame.SRCALPHA)
            pygame.draw.circle(s, (*GREEN, int(90 * pulse)),
                               (cx, cy), face_r + int(6 * pulse), 2)
            screen.blit(s, (0, 0))

        elif self.state == PROCESSING:
            for i in range(8):
                ang = now * 3 + i * (3.14159 / 4)
                sx  = cx + int((face_r + 12) * np.cos(ang))
                sy  = cy + int((face_r + 12) * np.sin(ang))
                pygame.draw.circle(screen, (*BLUE, 30 * i), (sx, sy), 5)

        # ── Status label ──────────────────────────────────────────────────
        if not self.models_ready:
            label, color = self.loading_msg, BLUE
        else:
            label, color = {
                IDLE:       ("",                         GREY),
                LISTENING:  ("MANTÉN ESPACIO PARA HABLAR", GREEN),
                PROCESSING: ("Pensando...",               BLUE),
                SPEAKING:   ("Jordi habla...",            ORANGE),
            }.get(self.state, ("", GREY))

        lbl = fonts["sm"].render(label, True, color)
        screen.blit(lbl, lbl.get_rect(center=(WIN_W // 2, WIN_H - 95)))

        # ── Subtitles ─────────────────────────────────────────────────────
        if self.subtitle:
            lines = wrap(self.subtitle, fonts["sub"], WIN_W - 60)[-3:]
            y = WIN_H - 75
            for line in lines:
                s = fonts["sub"].render(line, True, WHITE)
                screen.blit(s, s.get_rect(center=(WIN_W // 2, y)))
                y += 26

        pygame.display.flip()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--topic", default="tu vida y carrera")
    parser.add_argument("--photo", default=None)
    parser.add_argument("--tts", choices=["edge", "xtts"], default="edge",
                        help="TTS engine: edge (fast, online) or xtts (local clone, slow)")
    args = parser.parse_args()

    pygame.init()
    screen = pygame.display.set_mode((WIN_W, WIN_H))
    pygame.display.set_caption("The Wild Project — Jordi Wild")
    clock  = pygame.time.Clock()

    fonts = {
        "big": pygame.font.SysFont("DejaVu Sans", 46, bold=True),
        "sm":  pygame.font.SysFont("DejaVu Sans", 17),
        "sub": pygame.font.SysFont("DejaVu Sans", 19),
    }

    avatar = JordiAvatar(topic=args.topic, photo_path=args.photo)
    avatar.tts_mode = args.tts

    with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype=np.float32,
                        callback=avatar.audio_callback, blocksize=1024):
        space_held = False
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit(); sys.exit()
                elif event.type == pygame.KEYDOWN:
                    if event.key in (pygame.K_ESCAPE, pygame.K_q):
                        pygame.quit(); sys.exit()
                    elif event.key == pygame.K_SPACE and not space_held:
                        if avatar.models_ready and avatar.state == LISTENING:
                            space_held = True
                            avatar.start_recording()
                elif event.type == pygame.KEYUP:
                    if event.key == pygame.K_SPACE and space_held:
                        space_held = False
                        avatar.stop_recording()

            avatar.draw(screen, fonts)
            clock.tick(FPS)

if __name__ == "__main__":
    main()
