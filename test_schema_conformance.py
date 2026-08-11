"""ask_json's required-field check.

Run: cd diffusionMTUs && PYTHONPATH=. venv/bin/python test_schema_conformance.py

The other test files replace the whole LLM with a duck-typed fake, which means
the REAL `ask_json` has never been exercised by a test. Here the object is built
through the real `__init__` and only the transport (`ask`) is swapped, so the
code under test is the shipped one. `VLLM_BASE_URL`/`VLLM_API_KEY` are set
because `_make_client` honours them and they skip the api-key-file read; no
network happens at construction and `ask` is replaced before any call.
"""
import asyncio
import dataclasses
import json
import os

os.environ["VLLM_BASE_URL"] = "http://127.0.0.1:1/v1"   # never contacted
os.environ["VLLM_API_KEY"] = "test"

from config import Config
from llm import LLM, missing_required

PASS = FAIL = 0


def check(cond, label):
    global PASS, FAIL
    if cond:
        PASS += 1
    else:
        FAIL += 1
        print(f"  FAIL: {label}")


def make_llm(replies):
    """Real LLM, fake transport. `replies` is consumed in CALL ORDER — never
    keyed on prompt text, because the retry prompt quotes the problem back and
    a text-keyed fake would let the stage fix itself and measure nothing."""
    cfg = dataclasses.replace(Config(), worker_model="gemma-4-31b",
                              concurrency=1)
    llm = LLM(cfg)
    llm.seen_prompts = []
    it = iter(replies)

    async def fake_ask(prompt, system=None, max_tokens=None, schema=None):
        llm.seen_prompts.append(prompt)
        return next(it)

    llm.ask = fake_ask
    return llm


SCHEMA = {"type": "object",
          "properties": {"action": {"type": "string"},
                         "why": {"type": "string"}},
          "required": ["action", "why"]}


async def main():
    # ---- missing_required, directly. Known-good AND known-bad both, because a
    # detector that rejects everything looks exactly like one that works.
    check(missing_required({"action": "a", "why": "b"}, SCHEMA) == [],
          "conforming object -> no missing fields")
    check(missing_required({"action": "a"}, SCHEMA) == ["why"],
          "omitted field is reported")
    check(missing_required({}, SCHEMA) == ["action", "why"],
          "both omitted -> both reported, in schema order")
    check(missing_required({"action": "a", "why": None}, SCHEMA) == [],
          "present-but-null counts as present (required is about presence)")
    check(missing_required({"action": "a", "extra": 1}, SCHEMA) == ["why"],
          "extra fields are not violations")

    # things it must refuse to judge rather than false-positive on
    check(missing_required({"a": 1}, None) == [], "no schema -> nothing missing")
    check(missing_required({"a": 1}, {"type": "object"}) == [],
          "schema without `required` -> nothing missing")
    check(missing_required([1, 2], SCHEMA) == [],
          "array reply is not judged against an object schema")
    check(missing_required("text", SCHEMA) == [], "non-object reply not judged")
    check(missing_required({"a": 1}, {"required": "why"}) == [],
          "malformed `required` (not a list) is ignored, not crashed on")

    # ---- the real Vertex replies, 2026-08-11. Both models omitted `why` and
    # substituted a near-miss name; this is the regression this check exists for.
    opus = {"action": "ask", "target": {}, "reason": "...", "questions": [],
            "fallback": {}, "confidence": 0.6}
    sonnet = {"action": "patch", "reasoning": "...", "next_steps": []}
    check(missing_required(opus, SCHEMA) == ["why"],
          "recorded opus-5 reply is caught")
    check(missing_required(sonnet, SCHEMA) == ["why"],
          "recorded sonnet-5 reply is caught")

    # ---- ask_json: the success path, not just the failure path.
    llm = make_llm([json.dumps({"action": "patch", "why": "ok"})])
    got = await llm.ask_json("p", schema=SCHEMA)
    check(got == {"action": "patch", "why": "ok"}, "conforming reply returned")
    check(len(llm.seen_prompts) == 1, "conforming reply costs exactly one call")
    check(llm.schema_violations == 0, "conforming reply is not counted a violation")

    # ---- retry recovers, and the retry prompt names the field
    llm = make_llm([json.dumps({"action": "patch"}),
                    json.dumps({"action": "patch", "why": "now here"})])
    got = await llm.ask_json("p", schema=SCHEMA)
    check(got == {"action": "patch", "why": "now here"}, "retry recovers")
    check(len(llm.seen_prompts) == 2, "recovery took exactly one retry")
    check("why" in llm.seen_prompts[1],
          "retry prompt names the missing field")
    check(llm.schema_violations == 0, "a recovered reply is not a violation")

    # ---- exhausted: RETURNED, not raised, and counted
    llm = make_llm([json.dumps({"action": "patch"})] * 3)
    got = await llm.ask_json("p", schema=SCHEMA)
    check(got == {"action": "patch"},
          "non-conforming reply is returned, not raised (consumers use .get())")
    check(llm.schema_violations == 1, "exhausted non-conformance is counted")
    check(len(llm.seen_prompts) == 3, "retries+1 calls, no more")
    check(llm.usage_record()["schema_violations"] == 1,
          "the counter reaches the usage record")

    # ---- unparseable behaviour is unchanged by this fix
    llm = make_llm(["not json at all", json.dumps({"action": "a", "why": "b"})])
    got = await llm.ask_json("p", schema=SCHEMA)
    check(got == {"action": "a", "why": "b"}, "unparseable then good still works")
    check(llm.schema_violations == 0, "a parse retry is not a schema violation")

    llm = make_llm(["nope"] * 3)
    try:
        await llm.ask_json("p", schema=SCHEMA)
        check(False, "all-unparseable still raises ValueError")
    except ValueError:
        check(True, "all-unparseable still raises ValueError")
    check(llm.schema_violations == 0,
          "a raise is not also counted as a violation")

    # ---- no schema: must not reject anything
    llm = make_llm([json.dumps({"anything": 1})])
    got = await llm.ask_json("p")
    check(got == {"anything": 1}, "no schema -> reply passes through")
    check(len(llm.seen_prompts) == 1, "no schema -> no retry")

    # ---- an array reply against an object schema must not be retried forever
    llm = make_llm([json.dumps([1, 2, 3])])
    got = await llm.ask_json("p", schema=SCHEMA)
    check(got == [1, 2, 3], "array reply passes through unjudged")
    check(len(llm.seen_prompts) == 1, "array reply is not retried")

    print(f"\n{PASS} passed, {FAIL} failed")
    print("ALL PASS" if FAIL == 0 else "FAILURES")
    return 1 if FAIL else 0


raise SystemExit(asyncio.run(main()))
