"""Standalone Chat Completions adapter for the frozen DeepSeek experiment."""
import json
import threading
import time
from urllib.parse import urlsplit


class CloudProvider:
    def __init__(self, settings, max_calls=300):
        from openai import OpenAI
        self.client = OpenAI(base_url=settings["base_url"], api_key=settings["key"],
                             timeout=60, max_retries=0)
        self.model = settings["model"]
        self.thinking_disabled = (urlsplit(settings["base_url"]).hostname == "api.deepseek.com"
                                  and self.model in ("deepseek-flash", "deepseek-pro"))
        self.max_calls, self.calls = max_calls, []
        self.lock = threading.Lock()
        self.budget_rejections = 0

    def call(self, role, payload, messages):
        with self.lock:
            if len(self.calls) >= self.max_calls:
                self.budget_rejections += 1
                raise RuntimeError("API call budget exhausted")
            record = dict(role=role, status="pending", input_tokens=0, output_tokens=0)
            self.calls.append(record)
        started = time.perf_counter()
        try:
            kwargs = dict(model=self.model, messages=messages, temperature=0,
                          max_tokens=700, response_format={"type": "json_object"})
            if self.thinking_disabled:
                kwargs["extra_body"] = {"thinking": {"type": "disabled"}}
            response = self.client.chat.completions.create(**kwargs)
            if response.usage:
                record.update(input_tokens=response.usage.prompt_tokens,
                              output_tokens=response.usage.completion_tokens)
            reply = response.choices[0]
            record.update(status="success", finish_reason=reply.finish_reason,
                          output_truncated=reply.finish_reason == "length",
                          thinking_disabled=self.thinking_disabled)
            return json.loads(reply.message.content)
        except Exception as exc:
            record.update(status="failed", error_type=type(exc).__name__)
            raise
        finally:
            record["seconds"] = time.perf_counter() - started

    def summary(self):
        with self.lock:
            records = [dict(record) for record in self.calls]
        return dict(backend="api", simulated=False, calls=len(records), model_id=self.model,
                    thinking_disabled=self.thinking_disabled,
                    input_tokens=sum(record["input_tokens"] for record in records),
                    output_tokens=sum(record["output_tokens"] for record in records),
                    failures=sum(record["status"] == "failed" for record in records),
                    pending=sum(record["status"] == "pending" for record in records),
                    total_call_seconds=sum(record.get("seconds", 0) for record in records),
                    retries=0, budget_rejections=self.budget_rejections, records=records,
                    pricing="not supplied; no invented currency cost")
