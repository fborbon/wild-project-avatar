"""Drop-in replacement for `anthropic.Anthropic`, backed by Amazon Nova on AWS Bedrock instead
of the Anthropic API — the Anthropic Console key this repo used is shared across several other
projects and ran out of credit. Auth is via ambient AWS credentials (IAM role, or `aws configure`
for local dev) — no API key needed, billed through AWS instead.

Named `Anthropic` and exposing the same `.messages.create(model=, max_tokens=, messages=,
system=)` surface as the real SDK — including a top-level `.stop_reason` (jordi_avatar_voice.py
reads it in its error-handling path) — so call sites only need to change their import, not their
logic. `messages` is the full multi-turn conversation history, same shape the Anthropic SDK
expects: `[{"role": "user"|"assistant", "content": "..."}, ...]`.
"""

from __future__ import annotations

import json

NOVA_MODEL_ID = "eu.amazon.nova-lite-v1:0"
BEDROCK_REGION = "eu-west-1"


class _ContentBlock:
    def __init__(self, text: str) -> None:
        self.text = text


class _Message:
    def __init__(self, text: str, stop_reason: str) -> None:
        self.content = [_ContentBlock(text)]
        self.stop_reason = stop_reason


class Anthropic:
    def __init__(self, model_id: str = NOVA_MODEL_ID, region: str = BEDROCK_REGION) -> None:
        self._model_id = model_id
        self._region = region
        self._bedrock = None

    @property
    def messages(self) -> Anthropic:
        return self

    def create(
        self, model: str, max_tokens: int, messages: list[dict], system: str | None = None,
        **_ignored,
    ) -> _Message:
        if self._bedrock is None:
            import boto3
            self._bedrock = boto3.client("bedrock-runtime", region_name=self._region)

        body: dict = {
            "messages": [
                {"role": m["role"], "content": [{"text": m["content"]}]} for m in messages
            ],
            "inferenceConfig": {"maxTokens": max_tokens},
        }
        if system:
            body["system"] = [{"text": system}]

        response = self._bedrock.invoke_model(
            modelId=self._model_id, body=json.dumps(body),
            contentType="application/json", accept="application/json",
        )
        result = json.loads(response["body"].read())
        return _Message(
            text=result["output"]["message"]["content"][0]["text"],
            stop_reason=result.get("stopReason", ""),
        )
