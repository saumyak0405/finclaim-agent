"""Typed tool protocol, registry and backends.

Every tool has a versioned JSON-ish schema. The registry validates the
model's arguments *before* execution, enforces a timeout, scans the output
for prompt-injection patterns, and converts it into an `Evidence` record.

Backends are swappable:
- `LiveBackend`    real data (yfinance, SEC EDGAR, Google News RSS)
- `FixtureBackend` recorded responses -> deterministic, offline evals
- `FaultInjector`  wraps any backend to inject errors / timeouts
"""
from __future__ import annotations

import ast
import concurrent.futures as cf
import json
import operator
import random
import re
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Protocol


class ToolError(RuntimeError):
    pass


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    params: dict[str, dict[str, Any]]  # name -> {"type": "string"|"integer"|..., "required": bool, "enum": [...]}
    version: str = "1"
    timeout_s: float = 20.0

    def render(self) -> str:
        ps = []
        for k, v in self.params.items():
            extra = f" one of {v['enum']}" if "enum" in v else ""
            ps.append(f"{k}: {v['type']}{'' if v.get('required', True) else ' (optional)'}{extra}")
        return f"- {self.name}(v{self.version}): {self.description} Args: {', '.join(ps) or 'none'}"

    def validate(self, args: Any) -> str | None:
        if not isinstance(args, dict):
            return "args must be an object"
        unknown = set(args) - set(self.params)
        if unknown:
            return f"unknown args {sorted(unknown)} for {self.name}"
        types = {"string": str, "integer": int, "number": (int, float), "boolean": bool}
        for k, v in self.params.items():
            if k not in args:
                if v.get("required", True):
                    return f"missing required arg '{k}' for {self.name}"
                continue
            if not isinstance(args[k], types[v["type"]]):
                return f"arg '{k}' must be {v['type']}"
            if "enum" in v and args[k] not in v["enum"]:
                return f"arg '{k}' must be one of {v['enum']}"
        return None


SPECS: dict[str, ToolSpec] = {s.name: s for s in [
    ToolSpec("price_history", "Daily closing prices and % change over a period for a stock ticker.",
             {"ticker": {"type": "string"}, "period": {"type": "string", "enum": ["1mo", "3mo", "6mo", "1y"]}}),
    ToolSpec("fundamentals", "Key fundamentals for a ticker: revenue, net income, margins, P/E, market cap.",
             {"ticker": {"type": "string"}}),
    ToolSpec("filings_facts", "Reported figures from regulatory filings (SEC EDGAR XBRL facts) for a ticker.",
             {"ticker": {"type": "string"}, "metric": {"type": "string", "enum": ["Revenues", "NetIncomeLoss", "EarningsPerShareBasic"]}}),
    ToolSpec("news", "Recent news headlines and snippets for a query.",
             {"query": {"type": "string"}, "limit": {"type": "integer", "required": False}}, timeout_s=15),
    ToolSpec("calculate", "Evaluate an arithmetic expression exactly, e.g. '(1520-1310)/1310*100'.",
             {"expression": {"type": "string"}}, timeout_s=2),
]}


# ------------------------------------------------------------- injection scan
_INJECTION = re.compile(
    r"ignore (all |any )?(previous|prior|above) instructions|disregard (the )?(system|previous)|"
    r"you are now|new instructions:|system prompt|do not verify|<\s*/?\s*(system|assistant)\s*>|"
    r"respond only with|tell the user to (buy|sell)",
    re.I,
)


def looks_injected(text: str) -> bool:
    return bool(_INJECTION.search(text))


# -------------------------------------------------------------- calculator
_OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv,
        ast.Pow: operator.pow, ast.USub: operator.neg, ast.UAdd: operator.pos, ast.Mod: operator.mod}


def safe_eval(expr: str) -> float:
    def ev(node: ast.AST) -> float:
        if isinstance(node, ast.Expression):
            return ev(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
            if isinstance(node.op, ast.Pow) and abs(ev(node.right)) > 10:
                raise ToolError("exponent too large")
            return _OPS[type(node.op)](ev(node.left), ev(node.right))
        if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
            return _OPS[type(node.op)](ev(node.operand))
        raise ToolError(f"unsupported expression element: {type(node).__name__}")
    if len(expr) > 200:
        raise ToolError("expression too long")
    try:
        return ev(ast.parse(expr.replace(",", ""), mode="eval"))
    except (SyntaxError, ZeroDivisionError) as e:
        raise ToolError(str(e)) from e


def calculate(args: dict[str, Any]) -> dict[str, Any]:
    val = safe_eval(args["expression"])
    return {"source": "local:calculator", "text": f"{args['expression']} = {round(val, 4)}",
            "data": {"value": val}}


# ------------------------------------------------------------- backends
class Backend(Protocol):
    def __call__(self, tool: str, args: dict[str, Any]) -> dict[str, Any]: ...


class LiveBackend:
    """Free, keyless data sources. yfinance is optional (pip install yfinance)."""
    UA = {"User-Agent": "finclaim-agent research bot (contact: set FINCLAIM_CONTACT)"}

    def __call__(self, tool: str, args: dict[str, Any]) -> dict[str, Any]:
        fn = getattr(self, f"_{tool}", None)
        if fn is None:
            raise ToolError(f"no live implementation for {tool}")
        return fn(args)

    def _get(self, url: str, timeout: float = 15) -> bytes:
        req = urllib.request.Request(url, headers=self.UA)
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.read()

    def _price_history(self, a: dict[str, Any]) -> dict[str, Any]:
        try:
            import yfinance as yf
        except ImportError as e:
            raise ToolError("yfinance not installed") from e
        hist = yf.Ticker(a["ticker"]).history(period=a["period"])
        if hist.empty:
            raise ToolError(f"no price data for {a['ticker']}")
        first, last = float(hist["Close"].iloc[0]), float(hist["Close"].iloc[-1])
        chg = (last - first) / first * 100
        text = (f"{a['ticker']} close {hist.index[0].date()}: {first:.2f}; close {hist.index[-1].date()}: {last:.2f}; "
                f"change over {a['period']}: {chg:.2f}%; high {hist['High'].max():.2f}; low {hist['Low'].min():.2f}")
        return {"source": "yfinance", "text": text, "data": {"first": first, "last": last, "change_pct": chg}}

    def _fundamentals(self, a: dict[str, Any]) -> dict[str, Any]:
        try:
            import yfinance as yf
        except ImportError as e:
            raise ToolError("yfinance not installed") from e
        info = yf.Ticker(a["ticker"]).info or {}
        keys = ["longName", "totalRevenue", "netIncomeToCommon", "profitMargins", "trailingPE", "marketCap", "currency"]
        data = {k: info.get(k) for k in keys if info.get(k) is not None}
        if not data:
            raise ToolError(f"no fundamentals for {a['ticker']}")
        return {"source": "yfinance", "text": "; ".join(f"{k}: {v}" for k, v in data.items()), "data": data}

    def _filings_facts(self, a: dict[str, Any]) -> dict[str, Any]:
        tickers = json.loads(self._get("https://www.sec.gov/files/company_tickers.json"))
        cik = next((v["cik_str"] for v in tickers.values() if v["ticker"].upper() == a["ticker"].upper()), None)
        if cik is None:
            raise ToolError(f"{a['ticker']} not found in SEC ticker map (non-US listing?)")
        facts = json.loads(self._get(f"https://data.sec.gov/api/xbrl/companyfacts/CIK{int(cik):010d}.json"))
        units = facts["facts"]["us-gaap"].get(a["metric"], {}).get("units", {})
        rows = [r for u in units.values() for r in u if r.get("form") == "10-K" and r.get("fp") == "FY"]
        rows = sorted({(r["fy"], r["val"]) for r in rows})[-4:]
        if not rows:
            raise ToolError(f"no annual {a['metric']} facts for {a['ticker']}")
        text = f"{a['ticker']} {a['metric']} (10-K, fiscal year): " + "; ".join(f"FY{fy}: {val:,}" for fy, val in rows)
        return {"source": f"sec.gov CIK{cik}", "text": text, "data": {"rows": rows}}

    def _news(self, a: dict[str, Any]) -> dict[str, Any]:
        q = urllib.parse.quote(a["query"])
        root = ET.fromstring(self._get(f"https://news.google.com/rss/search?q={q}&hl=en-IN&gl=IN&ceid=IN:en"))
        items = root.findall(".//item")[: a.get("limit", 5)]
        if not items:
            raise ToolError("no news found")
        lines = [f"{i.findtext('pubDate', '')[:16]} | {i.findtext('title', '')}" for i in items]
        return {"source": "news.google.com rss", "text": "\n".join(lines), "data": {"count": len(lines)}}


@dataclass
class FixtureBackend:
    """Recorded tool responses. Matches on tool name + case-insensitive arg subset."""
    fixtures: list[dict[str, Any]]
    misses: list[tuple[str, dict[str, Any]]] = field(default_factory=list)

    @classmethod
    def load(cls, path: Path | str) -> "FixtureBackend":
        return cls(json.loads(Path(path).read_text())["calls"])

    def __call__(self, tool: str, args: dict[str, Any]) -> dict[str, Any]:
        norm = {k: str(v).lower() for k, v in args.items()}
        for fx in self.fixtures:
            if fx["tool"] != tool:
                continue
            if all(norm.get(k) == str(v).lower() for k, v in fx.get("match", {}).items()):
                if "error" in fx:
                    raise ToolError(fx["error"])
                return fx["result"]
        self.misses.append((tool, args))
        raise ToolError(f"no data available for {tool}({args})")


@dataclass
class FaultInjector:
    inner: Backend
    fail_rate: float = 0.0
    seed: int = 0
    rng: random.Random = field(init=False)

    def __post_init__(self) -> None:
        self.rng = random.Random(self.seed)

    def __call__(self, tool: str, args: dict[str, Any]) -> dict[str, Any]:
        if tool != "calculate" and self.rng.random() < self.fail_rate:
            raise ToolError(self.rng.choice(["upstream timeout", "HTTP 503 from provider", "rate limited (429)"]))
        return self.inner(tool, args)


# ------------------------------------------------------------- registry
@dataclass
class ToolOutcome:
    ok: bool
    result: dict[str, Any] | None = None
    error: str | None = None
    injected: bool = False


@dataclass
class ToolRegistry:
    backend: Backend
    specs: dict[str, ToolSpec] = field(default_factory=lambda: dict(SPECS))
    _pool: cf.ThreadPoolExecutor = field(default_factory=lambda: cf.ThreadPoolExecutor(max_workers=4), repr=False)

    def catalog(self) -> str:
        return "\n".join(s.render() for s in self.specs.values())

    def validate(self, tool: str, args: Any) -> str | None:
        if tool not in self.specs:
            return f"unknown tool '{tool}'. Available: {sorted(self.specs)}"
        return self.specs[tool].validate(args)

    def execute(self, tool: str, args: dict[str, Any]) -> ToolOutcome:
        err = self.validate(tool, args)
        if err:
            return ToolOutcome(False, error=f"validation: {err}")
        spec = self.specs[tool]
        fn: Callable[[], dict[str, Any]] = (lambda: calculate(args)) if tool == "calculate" else (lambda: self.backend(tool, args))
        fut = self._pool.submit(fn)
        try:
            result = fut.result(timeout=spec.timeout_s)
        except cf.TimeoutError:
            return ToolOutcome(False, error=f"timeout after {spec.timeout_s}s")
        except ToolError as e:
            return ToolOutcome(False, error=str(e))
        except Exception as e:  # never let a tool crash the loop
            return ToolOutcome(False, error=f"{type(e).__name__}: {e}")
        text = str(result.get("text", ""))[:4000]  # bound context growth
        result = {**result, "text": text}
        return ToolOutcome(True, result=result, injected=looks_injected(text))
