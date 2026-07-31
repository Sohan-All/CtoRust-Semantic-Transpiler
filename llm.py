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
    # max_retries=0 is deliberate and load-bearing. It was 5, which made
    # `timeout` a per-ATTEMPT bound and the real bound 6x that — 90 minutes for
    # a nominal 900s, on the very line whose comment records cutting 3600 to
    # 900 to stop a wedged request holding a run for an hour. Worse, the
    # client's internal retries emit nothing: no log line, no counter, so from
    # inside a run a 90-minute wedge and a slow call look identical. That is
    # what killed `array_list` t4 srvA.
    #
    # The outer `_TRANSIENT` loop in `ask` is now the SINGLE retry mechanism.
    # It already does exponential backoff, holds the sleep inside the semaphore
    # on purpose, and counts every retry into `transport_retries`. One retry
    # layer that reports beats two where the lower one is mute.
    #
    # `timeout` here is only a default; `ask` sets a per-request timeout scaled
    # to the token budget (see `_request_timeout`).
    return AsyncOpenAI(base_url=base_url, api_key=api_key,
                       timeout=timeout, max_retries=0)


# A 31B on two GPUs decodes at roughly 15-34 tok/s depending on load; 10 is a
# deliberate floor, not an estimate. Used to size a request's wall clock.
_MIN_DECODE_RATE = 10.0
# Connect, prefill and queueing slack, on top of the decode allowance.
_PREFILL_ALLOWANCE = 180.0


def _request_timeout(budget: int, configured: float) -> float:
    """Wall clock for ONE attempt, scaled to what that attempt may generate.

    A single fixed bound cannot serve both ends of the budget range. 900s is
    generous for the 512-4000 token calls that make up most of a run, and
    simultaneously UNREACHABLE for a 16000-token one: at the slow end of the
    observed decode rate that needs ~1070s, so it could never finish inside
    900s no matter how healthy the server. `MAX_TOKENS_CEILING` and
    `request_timeout` were mutually unsatisfiable, which is the trap
    `array_list` t4 srvA fell into after its budget doubled to the ceiling.

    Never shrinks below the configured value, so this only ever grants more
    time than before, and only in proportion to the tokens actually allowed.
    """
    return max(configured, _PREFILL_ALLOWANCE + budget / _MIN_DECODE_RATE)


# Transient by nature: the server is unreachable, overloaded, or restarting.
# The client itself no longer retries (max_retries=0), so the loop in `ask` is
# the only retry layer — which is what makes every retry appear in the log and
# in `transport_retries`.
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
        configured = float(getattr(self.cfg, "request_timeout", 900))
        # One logical ask() can legitimately issue several requests: the budget
        # doubles on a truncated reply, and a transient failure retries. Each is
        # individually bounded, but nothing bounded the PRODUCT — which is how a
        # single call held a run for 38 minutes with nothing in the log. This is
        # the backstop; exceeding it is a real failure and is raised, not
        # swallowed, so gather_units records it and the run scores nothing.
        deadline = asyncio.get_running_loop().time() + float(
            getattr(self.cfg, "call_deadline", 2700))
        attempt = 0
        while True:
            if asyncio.get_running_loop().time() > deadline:
                raise TimeoutError(
                    f"call exceeded call_deadline "
                    f"({getattr(self.cfg, 'call_deadline', 2700)}s) after "
                    f"{attempt} transport retries at budget {budget}")
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
                        timeout=_request_timeout(budget, configured),
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
