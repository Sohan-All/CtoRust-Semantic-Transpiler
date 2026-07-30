"""Thin async wrapper over a local vLLM server (OpenAI-compatible API).

Every pipeline operation is a fresh, stateless call — no conversations anywhere.
This module is the single seam where a different backend would plug in later.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
from pathlib import Path

import openai
from openai import AsyncOpenAI

from config import Config, MODEL_SERVERS


def _make_client(model: str, timeout: int = 900) -> AsyncOpenAI:
    """Resolve which vLLM instance serves `model` via MODEL_SERVERS, unless
    VLLM_BASE_URL/VLLM_API_KEY are set — those still win, for one-off manual
    overrides."""
    base_url = os.environ.get("VLLM_BASE_URL")
    api_key = os.environ.get("VLLM_API_KEY")
    if not base_url or not api_key:
        server = MODEL_SERVERS.get(model)
        if server is None:
            raise SystemExit(
                f"Unknown model {model!r}: no entry in config.MODEL_SERVERS "
                f"and no VLLM_BASE_URL/VLLM_API_KEY override set. Known "
                f"models: {', '.join(MODEL_SERVERS)}")
        base_url = base_url or server["base_url"]
        if not api_key:
            key_file = Path(server["api_key_file"])
            if not key_file.exists():
                raise SystemExit(f"No credentials: set VLLM_API_KEY or "
                                 f"create {key_file}.")
            api_key = key_file.read_text().strip()
    # Generous but bounded: a 31B model emitting a long reply under a queued
    # batch legitimately takes minutes, so this cannot be tight — but it was
    # 3600, which meant a single wedged request held a run open for an hour
    # before failing it. cfg.request_timeout.
    return AsyncOpenAI(base_url=base_url, api_key=api_key,
                       timeout=timeout, max_retries=5)


# Transient by nature: the server is unreachable, overloaded, or restarting.
# The OpenAI client already retries these `max_retries` times; the outer loop
# in `ask` exists for the case it cannot cover — a vLLM instance being
# restarted, which is down for minutes and then perfectly healthy.
_TRANSIENT = (openai.APIConnectionError, openai.APITimeoutError,
              openai.InternalServerError, openai.RateLimitError)


class LLM:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.client = _make_client(cfg.worker_model,
                                   getattr(cfg, "request_timeout", 900))
        self.sem = asyncio.Semaphore(cfg.concurrency)
        self.calls = 0
        self.input_tokens = 0
        self.output_tokens = 0
        self.transport_retries = 0   # observability: how flaky was this run

    MAX_TOKENS_CEILING = 16000
    MIN_TOKENS_FLOOR = 512  # below this a context-length clamp gives up

    async def ask(self, prompt: str, system: str | None = None,
                  max_tokens: int | None = None,
                  schema: dict | None = None) -> str:
        """One fresh call. Returns the text of the response.

        If the reply is cut off by its token budget (finish_reason ==
        "length"), the call is retried with a doubled budget — truncated
        code replies produce unclosed-delimiter artifacts downstream that no
        amount of repair fixes. If prompt + budget overflows the server's
        context window (vLLM 400s instead of truncating), the budget is
        halved until the request fits."""
        budget = max_tokens or self.cfg.max_tokens
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        max_transport = getattr(self.cfg, "transport_retries", 3)
        backoff = getattr(self.cfg, "retry_backoff", 5.0)
        attempt = 0
        while True:
            async with self.sem:
                try:
                    extra = {"top_k": self.cfg.top_k}
                    if schema is not None:
                        # constrained decoding: the server may only emit tokens
                        # that keep the output conforming, so a malformed reply
                        # becomes impossible rather than merely unlikely. Lets
                        # sampling stay at the model's own recommended values.
                        extra["structured_outputs"] = {"json": schema}
                    resp = await self.client.chat.completions.create(
                        model=self.cfg.worker_model,
                        max_tokens=budget,
                        messages=messages,
                        temperature=self.cfg.temperature,
                        top_p=self.cfg.top_p,
                        # top_k is not an OpenAI field; vLLM accepts it here
                        extra_body=extra,
                    )
                except openai.BadRequestError as e:
                    if "maximum context length" in str(e) and budget > self.MIN_TOKENS_FLOOR:
                        budget = max(budget // 2, self.MIN_TOKENS_FLOOR)
                        continue
                    raise
                except _TRANSIENT as e:
                    # held outside the semaphore would be better, but sleeping
                    # inside it is deliberate: if the server is down, letting
                    # the other three slots pile straight into the same failure
                    # just burns the retry budget four times as fast.
                    if attempt >= max_transport:
                        raise
                    delay = backoff * (2 ** attempt)
                    attempt += 1
                    self.transport_retries += 1
                    print(f"[llm] {type(e).__name__} — retry {attempt}/"
                          f"{max_transport} in {delay:.0f}s")
                    await asyncio.sleep(delay)
                    continue
            self.calls += 1
            if resp.usage:
                self.input_tokens += resp.usage.prompt_tokens
                self.output_tokens += resp.usage.completion_tokens
            # vLLM can return a well-formed response carrying no choices (an
            # aborted or preempted request). Indexing [0] made that an
            # IndexError, which — before gather_units — killed the whole stage.
            # Treat it as transient: it is a server-side hiccup, not a bad
            # prompt, and the next draw normally succeeds.
            if not resp.choices:
                if attempt >= max_transport:
                    raise RuntimeError(
                        f"server returned no choices after {attempt} retries")
                attempt += 1
                self.transport_retries += 1
                print(f"[llm] empty choices — retry {attempt}/{max_transport}")
                continue
            choice = resp.choices[0]
            if choice.finish_reason != "length" or budget >= self.MAX_TOKENS_CEILING:
                return choice.message.content or ""
            budget = min(budget * 2, self.MAX_TOKENS_CEILING)

    async def ask_json(self, prompt: str, system: str | None = None,
                       max_tokens: int | None = None, retries: int = 2,
                       schema: dict | None = None):
        """Fresh call expected to return a JSON object/array. Extracts the first
        JSON value from the response (models sometimes wrap in ``` fences).

        Pass `schema` to constrain decoding to it. Worth doing wherever a
        malformed reply is expensive: an unparseable stage-S spec used to take
        down a whole project run through asyncio.gather, having already burnt
        the retries below. The retry path stays as a backstop for servers that
        do not support structured outputs."""
        last_err = None
        for _ in range(retries + 1):
            text = await self.ask(prompt, system=system, max_tokens=max_tokens,
                                  schema=schema)
            try:
                return extract_json(text)
            except ValueError as e:
                last_err = e
                prompt = (prompt + "\n\nYour previous reply was not valid JSON "
                          f"({e}). Reply with ONLY the JSON, no prose, no fences.")
        raise ValueError(f"no valid JSON after retries: {last_err}")

    def usage_record(self) -> dict:
        return {
            "type": "event", "event": "usage",
            "calls": self.calls,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "transport_retries": self.transport_retries,
        }


def extract_json(text: str):
    """Pull the first JSON object or array out of a model response."""
    text = text.strip()
    fence = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    if fence:
        text = fence.group(1).strip()
    # find the first { or [ and parse from there with a raw-decode
    for i, ch in enumerate(text):
        if ch in "{[":
            try:
                obj, _ = json.JSONDecoder().raw_decode(text[i:])
                return obj
            except json.JSONDecodeError as e:
                raise ValueError(str(e)) from e
    raise ValueError("no JSON object/array found in response")
