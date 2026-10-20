"""Day 35: one Streamlit page over the RAG pipeline (hybrid + rerank + the
version/year rule; SELF_CHECK off). Each visitor brings their own Anthropic key;
the container holds only PINECONE_API_KEY.

    streamlit run streamlit_app.py   (in Docker: docker compose up)
"""

import re
from dataclasses import dataclass, replace

import anthropic
import pandas as pd
import streamlit as st

from multimodal_rag.answer import answered, ask
from multimodal_rag.config import EXAMPLE_QUESTIONS, SESSION_QUERY_CAP
from multimodal_rag.contracts import Answer, Hit
from multimodal_rag.evaluate import live_scores, ragas_embeddings
from multimodal_rag.retrieve import QueryError, Retriever

QUOTE_CHARS = 200  # citation quotes shown under an answer are cut to this


@dataclass(frozen=True)
class Turn:
    """One question and what came back. `answer` is None and `error` is set when
    the query path raised QueryError. `live` = RAGAS (faithfulness, answer
    relevancy, cost_usd), filled after the answer is shown; None until then and
    for abstains, refusals and errors."""

    question: str
    answer: Answer | None
    hits: list[Hit]
    error: str | None = None
    live: tuple[float, float, float] | None = None


@st.cache_resource(show_spinner="Loading BGE, the reranker and BM25 (once)…")
def get_retriever() -> Retriever:
    """One Retriever per server process, shared by every session: it is ~1.2 GB
    of process memory (torch, BGE, the cross-encoder, BM25; measured), which
    would not fit once per visitor. 8 searches from 4 threads matched the
    sequential results (Day 35)."""
    return Retriever()


@st.cache_resource(show_spinner=False)
def get_embeddings():
    """One BGE copy for live answer relevancy, shared by every session."""
    return ragas_embeddings()


def run_turn(question: str, history: list[str], api_key: str) -> Turn:
    """Answer `question` with the visitor's key. `history` = the session's
    earlier questions, so a follow-up is rewritten to stand alone."""
    client = anthropic.Anthropic(api_key=api_key)
    try:
        ans, hits = ask(question, get_retriever(), client, history)
    except QueryError as e:
        return Turn(question, None, [], str(e))
    return Turn(question, ans, hits)


def as_text(s: str) -> str:
    """`s` escaped so Streamlit markdown shows it as written (slide text holds
    "*", "_", "$" and brackets)."""
    return re.sub(r"([\\`*_\[\]()#~<>$|])", r"\\\1", s)


def show_answer(t: Turn) -> None:
    """Left column: the answer with its citations, or the abstain, refusal or
    error."""
    st.markdown(f"#### {as_text(t.question)}")
    if t.error:
        st.error(f"{t.error}\n\nNothing was answered. Try again in a moment.")
        return
    if not answered(t.answer):
        st.info(t.answer.text)
        return
    # Claude writes markdown (bold, lists); only "$" is escaped: 26 chunks hold
    # two or more dollar amounts, which would render as LaTeX between them.
    st.markdown(t.answer.text.replace("$", "\\$"))
    lines = []  # Claude often cites the same passage for several claims
    for c in t.answer.citations:
        mark = " (AI description of the slide)" if c.generated else ""
        quote = " ".join(c.cited_text.split())  # slide quotes span many lines
        if len(quote) > QUOTE_CHARS:
            quote = quote[:QUOTE_CHARS].rsplit(" ", 1)[0] + " …"
        line = f"{c.doc_id}, p.{c.page + 1}{mark}: “{quote}”"
        if line not in lines:
            lines.append(line)
    st.markdown("**Citations**")
    for line in lines:
        st.caption(as_text(line))


def show_metrics(t: Turn) -> None:
    """Right column: this question's cost and latency, the pages Claude read,
    and the live RAGAS scores (or a spinner slot while they are computed)."""
    if t.error:
        return
    a = t.answer
    st.markdown("##### This question")
    cost, latency = st.columns(2)
    cost.metric("Answer cost", f"${a.cost_usd:.4f}")
    latency.metric("Latency", f"{a.latency_s:.1f} s")
    st.markdown("##### Pages Claude Haiku read")
    pages = pd.DataFrame(
        {
            "page": [h.chunk.page + 1 for h in t.hits],
            "relevance": [round(h.score, 2) for h in t.hits],
            "document": [h.chunk.doc_id for h in t.hits],
        }
    )
    st.dataframe(pages, hide_index=True, height=200)
    st.caption("Relevance: the reranker's score; higher is more relevant.")
    st.markdown("##### Answer quality (RAGAS)")
    if t.live:
        faith, rel, scoring = t.live
        f_col, r_col = st.columns(2)  # a third metric is cut off at this width
        f_col.metric("Faithfulness", f"{faith:.2f}")
        r_col.metric("Answer relevancy", f"{rel:.2f}")
        st.caption(
            "Faithfulness: share of the answer's claims the pages support. "
            "Answer relevancy: how directly it answers the question. Both 0–1. "
            f"Scoring cost: ${scoring:.4f}."
        )
    elif not answered(a):
        st.caption("No answer, so nothing to score.")


def show_turn(t: Turn) -> tuple:
    """One row: answer left, metrics right. Returns the right column."""
    left, right = st.columns([3, 2])
    with left:
        show_answer(t)
    with right:
        show_metrics(t)
    st.divider()
    return right


st.set_page_config(page_title="FDA Document Q&A", layout="wide")
state = st.session_state
state.setdefault("turns", [])
state.setdefault("n_queries", 0)

clicked = None
with st.sidebar:
    api_key = st.text_input(
        "Your Anthropic API key",
        type="password",
        key="api_key",
        help="Used for this session's Claude calls only; never stored.",
    )
    st.subheader("About")
    st.markdown(
        "Ask questions about FDA drug regulation. Answers come only from "
        "52 public FDA documents (2,313 pages): presentations by FDA's drug "
        "center from 2017 to 2024 and two books, on topics such as generic "
        "drug approvals, new drugs, biosimilars, opioids and drug shortages."
    )
    st.markdown(
        "Every answer cites the document and page it came from. If the "
        "documents don't cover a question, the app says so instead of guessing."
    )
    st.markdown(
        "Each answer is scored live with RAGAS, using Claude Haiku as the judge."
    )
    st.caption(
        "Not medical or regulatory advice. Documents: ViDoRe V3 Pharmaceuticals "
        "dataset, CC BY 4.0."
    )
    st.subheader("Try")
    for q in EXAMPLE_QUESTIONS:
        if st.button(q, use_container_width=True):
            clicked = q

st.title("FDA Document Q&A")
st.caption("Type a question below, or pick an example in the sidebar.")
for turn in state.turns:
    show_turn(turn)

question = st.chat_input("Ask about the FDA documents") or clicked
if question and not api_key:
    st.warning("Enter your Anthropic API key in the sidebar first.")
elif question and state.n_queries >= SESSION_QUERY_CAP:
    st.warning(f"This session's limit of {SESSION_QUERY_CAP} questions is used up.")
elif question:
    state.n_queries += 1
    history = [t.question for t in state.turns if t.error is None]
    with st.spinner("Searching and answering…"):
        turn = run_turn(question, history, api_key)
    state.turns.append(turn)
    right = show_turn(turn)
    if not turn.error and answered(turn.answer):
        with right, st.spinner("Scoring the answer…"):
            live = live_scores(
                question, turn.answer, turn.hits, api_key, get_embeddings()
            )
        state.turns[-1] = replace(turn, live=live)
        st.rerun()

st.sidebar.caption(f"Questions this session: {state.n_queries} of {SESSION_QUERY_CAP}")
