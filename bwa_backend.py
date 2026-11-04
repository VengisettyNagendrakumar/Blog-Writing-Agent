from __future__ import annotations

import base64
import json
import operator
import os
import re
from datetime import date, timedelta
from pathlib import Path
from typing import TypedDict, List, Optional, Literal, Annotated, TypeVar

from pydantic import BaseModel, Field, ValidationError
import requests

from langgraph.graph import StateGraph, START, END
from langgraph.types import Send

# from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage
from dotenv import load_dotenv

load_dotenv()

# ============================================================
# Blog Writer (Router → (Research?) → Orchestrator → Workers → ReducerWithImages)
# Patches image capability using your 3-node reducer flow:
#   merge_content -> decide_images -> generate_and_place_images
# ============================================================


# -----------------------------
# 1) Schemas
# -----------------------------
class Task(BaseModel):
    id: int
    title: str
    goal: str = Field(..., description="One sentence describing what the reader should do/understand.")
    bullets: List[str] = Field(..., min_length=3, max_length=6)
    target_words: int = Field(..., description="Target words (120–550).")

    tags: List[str] = Field(default_factory=list)
    requires_research: bool = False
    requires_citations: bool = False
    requires_code: bool = False


class Plan(BaseModel):
    blog_title: str
    audience: str
    tone: str
    blog_kind: Literal["explainer", "tutorial", "news_roundup", "comparison", "system_design"] = "explainer"
    constraints: List[str] = Field(default_factory=list)
    tasks: List[Task]


class EvidenceItem(BaseModel):
    title: str
    url: str
    published_at: Optional[str] = None  # ISO "YYYY-MM-DD" preferred
    snippet: Optional[str] = None
    source: Optional[str] = None


class RouterDecision(BaseModel):
    needs_research: bool
    mode: Literal["closed_book", "hybrid", "open_book"]
    reason: str
    queries: List[str] = Field(default_factory=list)
    max_results_per_query: int = Field(5)


class EvidencePack(BaseModel):
    evidence: List[EvidenceItem] = Field(default_factory=list)


# ---- Image planning schema (ported from your image flow) ----
class ImageSpec(BaseModel):
    placeholder: str = ""
    section_heading: str = Field(..., description="Exact Markdown heading where the image belongs.")
    filename: str = Field(..., description="Save under images/, e.g. qkv_flow.png")
    alt: str
    caption: str
    prompt: str = Field(..., description="Prompt to send to the image model.")
    size: Literal["1024x1024", "1024x1536", "1536x1024"] = "1024x1024"
    quality: Literal["low", "medium", "high"] = "medium"


class GlobalImagePlan(BaseModel):
    images: List[ImageSpec] = Field(default_factory=list)


class State(TypedDict):
    topic: str

    # routing / research
    mode: str
    needs_research: bool
    queries: List[str]
    evidence: List[EvidenceItem]
    plan: Optional[Plan]

    # recency
    as_of: str
    recency_days: int

    # workers
    sections: Annotated[List[tuple[int, str]], operator.add]  # (task_id, section_md)

    # reducer/image
    merged_md: str
    md_with_placeholders: str
    image_specs: List[dict]
    image_planning_warning: str
    image_generation_errors: List[str]

    final: str


# -----------------------------
# 2) LLM
# -----------------------------
# llm = ChatOpenAI(model="gpt-4.1-mini")
llm = ChatGroq(model="openai/gpt-oss-120b")
ModelT = TypeVar("ModelT", bound=BaseModel)


def _invoke_structured_output(
    schema: type[ModelT],
    messages: List[SystemMessage | HumanMessage],
) -> ModelT:
    system_message = messages[0]
    if not isinstance(system_message.content, str):
        raise ValueError("Structured-output system prompt must be plain text.")

    schema_json = json.dumps(schema.model_json_schema(), ensure_ascii=False)
    prompt = (
        f"{system_message.content}\n\n"
        "Return only one valid JSON object matching this JSON Schema. "
        "Do not use a tool call, markdown fences, or text outside the JSON object.\n"
        f"JSON Schema:\n{schema_json}"
    )
    response = llm.invoke([SystemMessage(content=prompt), *messages[1:]])
    if isinstance(response.content, str):
        response_text = response.content
    elif isinstance(response.content, list):
        response_text = "\n".join(
            item["text"]
            for item in response.content
            if isinstance(item, dict) and isinstance(item.get("text"), str)
        )
    else:
        response_text = ""

    decoder = json.JSONDecoder()
    parsed = None
    for start, character in enumerate(response_text):
        if character != "{":
            continue
        try:
            parsed, _ = decoder.raw_decode(response_text[start:])
            break
        except json.JSONDecodeError:
            continue
    if not isinstance(parsed, dict):
        raise ValueError(f"{schema.__name__} response did not contain a JSON object.")

    if (
        parsed.get("name") == schema.__name__
        and isinstance(parsed.get("arguments"), dict)
    ):
        parsed = parsed["arguments"]
    return schema.model_validate(parsed)


# -----------------------------
# 3) Router
# -----------------------------
ROUTER_SYSTEM = """You are a routing module for a technical blog planner.

Decide whether web research is needed BEFORE planning.

Modes:
- closed_book (needs_research=false): evergreen concepts.
- hybrid (needs_research=true): evergreen + needs up-to-date examples/tools/models.
- open_book (needs_research=true): volatile weekly/news/"latest"/pricing/policy.

If needs_research=true:
- Output 3–10 high-signal, scoped queries.
- For open_book weekly roundup, include queries reflecting last 7 days.
"""

def router_node(state: State) -> dict:
    decision = _invoke_structured_output(
        RouterDecision,
        [
            SystemMessage(content=ROUTER_SYSTEM),
            HumanMessage(content=f"Topic: {state['topic']}\nAs-of date: {state['as_of']}"),
        ],
    )

    if decision.mode == "open_book":
        recency_days = 7
    elif decision.mode == "hybrid":
        recency_days = 45
    else:
        recency_days = 3650

    return {
        "needs_research": decision.needs_research,
        "mode": decision.mode,
        "queries": decision.queries,
        "recency_days": recency_days,
    }

def route_next(state: State) -> str:
    return "research" if state["needs_research"] else "orchestrator"

# -----------------------------
# 4) Research (Tavily)
# -----------------------------
def _tavily_search(query: str, max_results: int = 5) -> List[dict]:
    if not os.getenv("TAVILY_API_KEY"):
        return []
    try:
        from langchain_community.tools.tavily_search import TavilySearchResults  # type: ignore
        tool = TavilySearchResults(max_results=max_results)
        results = tool.invoke({"query": query})
        out: List[dict] = []
        for r in results or []:
            out.append(
                {
                    "title": r.get("title") or "",
                    "url": r.get("url") or "",
                    "snippet": r.get("content") or r.get("snippet") or "",
                    "published_at": r.get("published_date") or r.get("published_at"),
                    "source": r.get("source"),
                }
            )
        return out
    except Exception:
        return []

def _iso_to_date(s: Optional[str]) -> Optional[date]:
    if not s:
        return None
    try:
        return date.fromisoformat(s[:10])
    except Exception:
        return None

RESEARCH_SYSTEM = """You are a research synthesizer.

Given raw web search results, produce EvidenceItem objects.

Rules:
- Only include items with a non-empty url.
- Prefer relevant + authoritative sources.
- Normalize published_at to ISO YYYY-MM-DD if reliably inferable; else null (do NOT guess).
- Keep snippets short.
- Deduplicate by URL.
"""

def research_node(state: State) -> dict:
    queries = (state.get("queries") or [])[:5]
    raw: List[dict] = []
    for q in queries:
        raw.extend(_tavily_search(q, max_results=3))

    if not raw:
        return {"evidence": []}

    raw = [
        {
            "title": str(result.get("title") or "")[:160],
            "url": str(result.get("url") or "")[:500],
            "snippet": str(result.get("snippet") or "")[:400],
            "published_at": str(result.get("published_at") or "")[:40],
            "source": str(result.get("source") or "")[:80],
        }
        for result in raw[:10]
    ]

    pack = _invoke_structured_output(
        EvidencePack,
        [
            SystemMessage(content=RESEARCH_SYSTEM),
            HumanMessage(
                content=(
                    f"As-of date: {state['as_of']}\n"
                    f"Recency days: {state['recency_days']}\n\n"
                    f"Raw results:\n{raw}"
                )
            ),
        ],
    )

    dedup = {}
    for e in pack.evidence:
        if e.url:
            dedup[e.url] = e
    evidence = list(dedup.values())

    if state.get("mode") == "open_book":
        as_of = date.fromisoformat(state["as_of"])
        cutoff = as_of - timedelta(days=int(state["recency_days"]))
        evidence = [e for e in evidence if (d := _iso_to_date(e.published_at)) and d >= cutoff]

    return {"evidence": evidence}

# -----------------------------
# 5) Orchestrator (Plan)
# -----------------------------
ORCH_SYSTEM = """You are a senior technical writer and developer advocate.
Produce a highly actionable outline for a technical blog post.

Requirements:
- 5–9 tasks, each with goal + 3–6 bullets + target_words.
- Tags are flexible; do not force a fixed taxonomy.

Grounding:
- closed_book: evergreen, no evidence dependence.
- hybrid: use evidence for up-to-date examples; mark those tasks requires_research=True and requires_citations=True.
- open_book: weekly/news roundup:
  - Set blog_kind="news_roundup"
  - No tutorial content unless requested
  - If evidence is weak, plan should explicitly reflect that (don’t invent events).

Output must match Plan schema.
"""

def orchestrator_node(state: State) -> dict:
    mode = state.get("mode", "closed_book")
    evidence = state.get("evidence", [])

    forced_kind = "news_roundup" if mode == "open_book" else None

    plan = _invoke_structured_output(
        Plan,
        [
            SystemMessage(content=ORCH_SYSTEM),
            HumanMessage(
                content=(
                    f"Topic: {state['topic']}\n"
                    f"Mode: {mode}\n"
                    f"As-of: {state['as_of']} (recency_days={state['recency_days']})\n"
                    f"{'Force blog_kind=news_roundup' if forced_kind else ''}\n\n"
                    f"Evidence:\n{[e.model_dump() for e in evidence][:16]}"
                )
            ),
        ],
    )
    if forced_kind:
        plan.blog_kind = "news_roundup"

    return {"plan": plan}


# -----------------------------
# 6) Fanout
# -----------------------------
def fanout(state: State):
    assert state["plan"] is not None
    return [
        Send(
            "worker",
            {
                "task": task.model_dump(),
                "topic": state["topic"],
                "mode": state["mode"],
                "as_of": state["as_of"],
                "recency_days": state["recency_days"],
                "plan": state["plan"].model_dump(),
                "evidence": [e.model_dump() for e in state.get("evidence", [])],
            },
        )
        for task in state["plan"].tasks
    ]

# -----------------------------
# 7) Worker
# -----------------------------
WORKER_SYSTEM = """You are a senior technical writer and developer advocate.
Write ONE section of a technical blog post in Markdown.

Constraints:
- Cover ALL bullets in order.
- Target words ±15%.
- Output only section markdown starting with "## <Section Title>".

Scope guard:
- If blog_kind=="news_roundup", do NOT drift into tutorials (scraping/RSS/how to fetch).
  Focus on events + implications.

Grounding:
- If mode=="open_book": do not introduce any specific event/company/model/funding/policy claim unless supported by provided Evidence URLs.
  For each supported claim, attach a Markdown link ([Source](URL)).
  If unsupported, write "Not found in provided sources."
- If requires_citations==true (hybrid tasks): cite Evidence URLs for external claims.

Code:
- If requires_code==true, include at least one minimal snippet.
"""

def worker_node(payload: dict) -> dict:
    task = Task(**payload["task"])
    plan = Plan(**payload["plan"])
    evidence = [EvidenceItem(**e) for e in payload.get("evidence", [])]

    bullets_text = "\n- " + "\n- ".join(task.bullets)
    evidence_text = "\n".join(
        f"- {e.title} | {e.url} | {e.published_at or 'date:unknown'}"
        for e in evidence[:20]
    )

    section_md = llm.invoke(
        [
            SystemMessage(content=WORKER_SYSTEM),
            HumanMessage(
                content=(
                    f"Blog title: {plan.blog_title}\n"
                    f"Audience: {plan.audience}\n"
                    f"Tone: {plan.tone}\n"
                    f"Blog kind: {plan.blog_kind}\n"
                    f"Constraints: {plan.constraints}\n"
                    f"Topic: {payload['topic']}\n"
                    f"Mode: {payload.get('mode')}\n"
                    f"As-of: {payload.get('as_of')} (recency_days={payload.get('recency_days')})\n\n"
                    f"Section title: {task.title}\n"
                    f"Goal: {task.goal}\n"
                    f"Target words: {task.target_words}\n"
                    f"Tags: {task.tags}\n"
                    f"requires_research: {task.requires_research}\n"
                    f"requires_citations: {task.requires_citations}\n"
                    f"requires_code: {task.requires_code}\n"
                    f"Bullets:{bullets_text}\n\n"
                    f"Evidence (ONLY cite these URLs):\n{evidence_text}\n"
                )
            ),
        ]
    ).content.strip()

    return {"sections": [(task.id, section_md)]}

# ============================================================
# 8) ReducerWithImages (subgraph)
#    merge_content -> decide_images -> generate_and_place_images
# ============================================================
def merge_content(state: State) -> dict:
    plan = state["plan"]
    if plan is None:
        raise ValueError("merge_content called without plan.")
    ordered_sections = [md for _, md in sorted(state["sections"], key=lambda x: x[0])]
    body = "\n\n".join(ordered_sections).strip()
    merged_md = f"# {plan.blog_title}\n\n{body}\n"
    return {"merged_md": merged_md}


DECIDE_IMAGES_SYSTEM = """You are an expert technical editor.
Decide if images/diagrams are needed for THIS blog.

Rules:
- Max 3 images total.
- Each image must materially improve understanding (diagram/flow/table-like visual).
- Choose section_heading exactly from the provided Markdown headings.
- Avoid decorative images; prefer technical diagrams with short labels.
Return only a JSON object with this shape:
{"images":[{"section_heading":"## Exact heading","filename":"diagram.png","alt":"Short description","caption":"Short caption","prompt":"Image-generation prompt","size":"1024x1024","quality":"medium"}]}
Use an empty images array if no image would help. Do not include markdown fences or explanatory text.
"""

def decide_images(state: State) -> dict:
    merged_md = state["merged_md"]
    plan = state["plan"]
    assert plan is not None
    headings = re.findall(r"^#{1,3}\s+.+$", merged_md, flags=re.MULTILINE)
    if not headings:
        return {
            "md_with_placeholders": merged_md,
            "image_specs": [],
            "image_planning_warning": "",
        }
    section_headings = [
        heading for heading in headings if not re.match(r"^#\s", heading)
    ] or headings

    try:
        image_plan = _invoke_structured_output(
            GlobalImagePlan,
            [
                SystemMessage(content=DECIDE_IMAGES_SYSTEM),
                HumanMessage(
                    content=(
                        f"Blog kind: {plan.blog_kind}\n"
                        f"Topic: {state['topic']}\n\n"
                        "Choose up to 3 useful images based on these section headings. "
                        "Do not return or rewrite the article. Return only image metadata.\n"
                        "Available headings:\n"
                        + "\n".join(section_headings)
                    )
                ),
            ],
        )
    except (ValueError, ValidationError):
        return {
            "md_with_placeholders": merged_md,
            "image_specs": [],
            "image_planning_warning": (
                "The image planner returned invalid image metadata, so the blog "
                "was generated without images. Try generating again to retry image planning."
            ),
        }

    image_specs = []
    placements = {}
    heading_by_normalized = {
        heading.lstrip("#").strip().casefold(): heading for heading in section_headings
    }
    for spec in image_plan.images[:3]:
        heading = heading_by_normalized.get(spec.section_heading.lstrip("#").strip().casefold())
        if heading is None:
            heading = section_headings[len(image_specs) % len(section_headings)]
        placeholder = f"[[IMAGE_{len(image_specs) + 1}]]"
        image_specs.append(
            {
                **spec.model_dump(),
                "placeholder": placeholder,
                "section_heading": heading,
                "filename": Path(spec.filename).name,
            }
        )
        placements.setdefault(heading, []).append(placeholder)

    markdown_lines = []
    for line in merged_md.splitlines():
        markdown_lines.append(line)
        for placeholder in placements.get(line, []):
            markdown_lines.append("")
            markdown_lines.append(placeholder)
    md_with_placeholders = "\n".join(markdown_lines)
    if merged_md.endswith("\n"):
        md_with_placeholders += "\n"

    return {
        "md_with_placeholders": md_with_placeholders,
        "image_specs": image_specs,
        "image_planning_warning": "",
    }


def _cloudflare_generate_image_bytes(prompt: str) -> bytes:
    """Generate an image using Cloudflare Workers AI FLUX.1 Schnell."""
    account_id = os.environ.get("CLOUDFLARE_ACCOUNT_ID")
    api_token = os.environ.get("CLOUDFLARE_API_TOKEN")
    if not account_id:
        raise RuntimeError("CLOUDFLARE_ACCOUNT_ID is not set.")
    if not api_token:
        raise RuntimeError("CLOUDFLARE_API_TOKEN is not set.")

    endpoint = (
        "https://api.cloudflare.com/client/v4/accounts/"
        f"{account_id}/ai/run/@cf/black-forest-labs/flux-1-schnell"
    )
    response = requests.post(
        endpoint,
        headers={"Authorization": f"Bearer {api_token}"},
        json={"prompt": prompt[:2048], "steps": 4},
        timeout=120,
    )
    if not response.ok:
        raise RuntimeError(
            f"Cloudflare Workers AI returned HTTP {response.status_code}: "
            f"{response.text[:1000]}"
        )

    try:
        image_base64 = response.json()["result"]["image"]
        return base64.b64decode(image_base64, validate=True)
    except (ValueError, KeyError, TypeError) as exc:
        raise RuntimeError(
            "Cloudflare Workers AI returned an invalid image response."
        ) from exc


# Previous Gemini image generation implementation (kept for reference):
#
# def _gemini_generate_image_bytes(prompt: str) -> bytes:
#     """
#     Returns raw image bytes generated by Gemini.
#     Requires: pip install google-genai
#     Env var: GOOGLE_API_KEY
#     """
#     from google import genai
#     from google.genai import types
#
#     api_key = os.environ.get("GOOGLE_API_KEY")
#     if not api_key:
#         raise RuntimeError("GOOGLE_API_KEY is not set.")
#
#     client = genai.Client(api_key=api_key)
#
#     resp = client.models.generate_content(
#         model="gemini-2.5-flash-image",
#         contents=prompt,
#         config=types.GenerateContentConfig(
#             response_modalities=["IMAGE"],
#             safety_settings=[
#                 types.SafetySetting(
#                     category="HARM_CATEGORY_DANGEROUS_CONTENT",
#                     threshold="BLOCK_ONLY_HIGH",
#                 )
#             ],
#         ),
#     )
#
#     parts = getattr(resp, "parts", None)
#     if not parts and getattr(resp, "candidates", None):
#         try:
#             parts = resp.candidates[0].content.parts
#         except Exception:
#             parts = None
#
#     if not parts:
#         raise RuntimeError("No image content returned (safety/quota/SDK change).")
#
#     for part in parts:
#         inline = getattr(part, "inline_data", None)
#         if inline and getattr(inline, "data", None):
#             return inline.data
#
#     raise RuntimeError("No inline image bytes found in response.")


def _safe_slug(title: str) -> str:
    s = title.strip().lower()
    s = re.sub(r"[^a-z0-9 _-]+", "", s)
    s = re.sub(r"\s+", "_", s).strip("_")
    return s or "blog"


def generate_and_place_images(state: State) -> dict:
    plan = state["plan"]
    assert plan is not None

    md = state.get("md_with_placeholders") or state["merged_md"]
    image_specs = state.get("image_specs", []) or []
    image_generation_errors: List[str] = []

    # If no images requested, just write merged markdown
    if not image_specs:
        filename = f"{_safe_slug(plan.blog_title)}.md"
        Path(filename).write_text(md, encoding="utf-8")
        return {"final": md, "image_generation_errors": image_generation_errors}

    images_dir = Path("images")
    images_dir.mkdir(exist_ok=True)

    for spec in image_specs:
        placeholder = spec["placeholder"]
        filename = spec["filename"]
        out_path = images_dir / filename

        # generate only if needed
        if not out_path.exists():
            try:
                img_bytes = _cloudflare_generate_image_bytes(spec["prompt"])
                out_path.write_bytes(img_bytes)
            except Exception as e:
                error_message = f"{filename}: {type(e).__name__}: {e}"
                image_generation_errors.append(error_message)
                # graceful fallback: keep doc usable
                prompt_block = (
                    f"> **[IMAGE GENERATION FAILED]** {spec.get('caption','')}\n>\n"
                    f"> **Alt:** {spec.get('alt','')}\n>\n"
                    f"> **Prompt:** {spec.get('prompt','')}\n>\n"
                    f"> **Error:** {error_message}\n"
                )
                md = md.replace(placeholder, prompt_block)
                continue

        img_md = f"![{spec['alt']}](images/{filename})\n*{spec['caption']}*"
        md = md.replace(placeholder, img_md)

    filename = f"{_safe_slug(plan.blog_title)}.md"
    Path(filename).write_text(md, encoding="utf-8")
    return {"final": md, "image_generation_errors": image_generation_errors}

# build reducer subgraph
reducer_graph = StateGraph(State)
reducer_graph.add_node("merge_content", merge_content)
reducer_graph.add_node("decide_images", decide_images)
reducer_graph.add_node("generate_and_place_images", generate_and_place_images)
reducer_graph.add_edge(START, "merge_content")
reducer_graph.add_edge("merge_content", "decide_images")
reducer_graph.add_edge("decide_images", "generate_and_place_images")
reducer_graph.add_edge("generate_and_place_images", END)
reducer_subgraph = reducer_graph.compile()

# -----------------------------
# 9) Build main graph
# -----------------------------
g = StateGraph(State)
g.add_node("router", router_node)
g.add_node("research", research_node)
g.add_node("orchestrator", orchestrator_node)
g.add_node("worker", worker_node)
g.add_node("reducer", reducer_subgraph)

g.add_edge(START, "router")
g.add_conditional_edges("router", route_next, {"research": "research", "orchestrator": "orchestrator"})
g.add_edge("research", "orchestrator")

g.add_conditional_edges("orchestrator", fanout, ["worker"])
g.add_edge("worker", "reducer")
g.add_edge("reducer", END)

app = g.compile()
app
