"""Query-path edge cases: abstain without Claude, citation verification, the
0-indexed page, early stop, Pinecone down, query transforms, BM25 + RRF,
reranking, citation metadata, a query with no keyword match, the self-check
retry. No network calls."""

from types import SimpleNamespace as NS

import pytest
from pinecone.errors import PineconeTimeoutError, RateLimitError

from multimodal_rag.answer import (
    answer,
    ask,
    build_documents,
    check_and_retry,
    verify_citations,
)
from multimodal_rag.config import ABSTAIN_TEXT, ANSWER_K, QUERY_PREFIX, REFUSAL_TEXT
from multimodal_rag.contracts import Answer, Chunk, Hit
from multimodal_rag.retrieve import QueryError, Retriever, rrf, search_query, tokenize

TEXT = "FDA approved 50 novel drugs in 2021. Most were small molecules."


def _hit(
    score: float = 0.8, page: int = 4, text: str = TEXT, kind: str = "text"
) -> Hit:
    return Hit(Chunk(f"deck-{page}-0", "deck", page, text, 12, kind), score)


def _cite(start: int, end: int, cited: str, doc: int = 0) -> NS:
    return NS(
        type="char_location",
        document_index=doc,
        start_char_index=start,
        end_char_index=end,
        cited_text=cited,
    )


class NoCallClient:
    """Fails the test if Claude is called."""

    @property
    def messages(self):
        raise AssertionError("Claude must not be called")


class FakeClient:
    """Returns one canned Messages API response."""

    def __init__(self, content: list, stop_reason: str = "end_turn") -> None:
        usage = NS(input_tokens=1000, output_tokens=100)
        self.response = NS(content=content, stop_reason=stop_reason, usage=usage)
        self.messages = self
        self.calls: list[dict] = []

    def create(self, **kwargs) -> NS:
        assert "temperature" not in kwargs  # Haiku 5.5 rejects it
        self.calls.append(kwargs)
        return self.response


def test_low_score_still_goes_to_claude() -> None:
    """No score threshold: query 218's gold page ranked 3rd at top-1 0.637."""
    content = [NS(type="text", text="50 drugs.", citations=[_cite(0, 36, TEXT[:36])])]
    result = answer("q", [_hit(score=0.30)], FakeClient(content))
    assert not result.abstained and result.cost_usd > 0


def test_no_hits_abstains_without_calling_claude() -> None:
    assert answer("q", [], NoCallClient()).abstained


def test_document_title_shows_1_indexed_page() -> None:
    """Page 4 in the data is the 5th PDF page a reader would open."""
    assert build_documents([_hit(page=4)])[0]["title"] == "deck p.5"


def test_citation_kept_only_if_quote_matches_chunk() -> None:
    hits = [_hit(page=4), _hit(page=9, text="Other text entirely.")]
    good = _cite(0, 36, TEXT[:36])
    wrong_span = _cite(0, 36, "FDA approved 60 novel drugs in 2021.")
    wrong_doc = _cite(0, 36, TEXT[:36], doc=1)
    content = [NS(type="text", text="50.", citations=[good, wrong_span, wrong_doc])]
    cites = verify_citations(content, hits)
    assert len(cites) == 1
    c = cites[0]
    assert (c.chunk_id, c.page, c.score) == ("deck-4-0", 4, 0.8)
    assert c.cited_text == "FDA approved 50 novel drugs in 2021." and not c.generated


def test_answer_with_citation_and_cost() -> None:
    content = [NS(type="text", text="50 drugs.", citations=[_cite(0, 36, TEXT[:36])])]
    result = answer("How many?", [_hit()], FakeClient(content))
    assert not result.abstained and len(result.citations) == 1
    assert result.cost_usd == pytest.approx((1000 * 0.10 + 100 * 0.50) / 1e6)


def test_claude_abstain_text_without_citations_is_abstain() -> None:
    content = [NS(type="text", text=ABSTAIN_TEXT, citations=None)]
    result = answer("Unrelated?", [_hit()], FakeClient(content))
    assert result.abstained and result.citations == []


def test_max_tokens_stop_raises() -> None:
    content = [NS(type="text", text="The answer is", citations=None)]
    with pytest.raises(QueryError, match="max_tokens"):
        answer("q", [_hit()], FakeClient(content, stop_reason="max_tokens"))


def test_vision_chunk_is_labelled_generated_for_claude_and_in_citations() -> None:
    hit = _hit(page=4, kind="vision")
    assert (
        build_documents([hit])[0]["title"] == "generated visual description, deck p.5"
    )
    content = [NS(type="text", text="50.", citations=[_cite(0, 36, TEXT[:36])])]
    assert verify_citations(content, [hit])[0].generated


def test_refusal_is_its_own_outcome() -> None:
    """Partial text and citations are discarded, cost is kept, not an abstain."""
    content = [NS(type="text", text="Partial", citations=[_cite(0, 36, TEXT[:36])])]
    result = answer("q", [_hit()], FakeClient(content, stop_reason="refusal"))
    assert result.refused and not result.abstained
    assert result.text == REFUSAL_TEXT and result.citations == []
    assert result.cost_usd > 0


def _retriever(
    index, strategy: str = "vector", chunks: list[Chunk] = (), reranker=None
) -> Retriever:
    r = object.__new__(Retriever)  # skip loading BGE and connecting to Pinecone
    r.strategy = strategy
    r.reranker = reranker
    r.model = NS(encode=lambda text, normalize_embeddings: NS(tolist=lambda: [0.0]))
    r.chunks = {c.chunk_id: c for c in [_hit().chunk, *chunks]}
    r.index = index
    if strategy == "hybrid":
        r.build_bm25()
    return r


def test_pinecone_timeout_becomes_query_error() -> None:
    def query(**kwargs):
        raise PineconeTimeoutError("timed out")

    with pytest.raises(QueryError, match="Pinecone unavailable"):
        _retriever(NS(query=query)).search("q")


def test_pinecone_rate_limit_becomes_query_error() -> None:
    """A 429 that outlasts the SDK's own retries reaches the app as QueryError,
    not as an uncaught exception."""

    def query(**kwargs):
        raise RateLimitError()

    with pytest.raises(QueryError, match="RateLimitError"):
        _retriever(NS(query=query)).search("q")


def test_ask_returns_the_hits_claude_read() -> None:
    """The app shows the pages and RAGAS scores the answer against them: the
    first ANSWER_K of the 20 searched."""
    chunks = [Chunk(f"deck-{p}-0", "deck", p, TEXT, 12, "text") for p in range(20)]
    matches = [NS(id=c.chunk_id, score=0.9 - p / 100) for p, c in enumerate(chunks)]
    r = _retriever(NS(query=lambda **kw: NS(matches=matches)), chunks=chunks)
    client = FakeClient([NS(type="text", text="50.", citations=None)])
    ans, hits = ask("q", r, client, self_check_on=False)
    assert ans.text == "50." and len(client.calls[0]["messages"][0]["content"]) == 11
    assert [h.chunk.page for h in hits] == list(range(ANSWER_K))


def test_id_missing_from_chunk_file_raises() -> None:
    """Index and chunks.jsonl from different ingest runs must not pass silently."""
    matches = [NS(id="deck-4-0", score=0.9), NS(id="deck-7-2", score=0.8)]
    index = NS(query=lambda **kwargs: NS(matches=matches))
    with pytest.raises(QueryError, match="deck-7-2"):
        _retriever(index).search("q")


def _reply(text: str) -> FakeClient:
    return FakeClient([NS(type="text", text=text, citations=None)])


def _query(text: str) -> FakeClient:
    """A forced record_query tool call, as the rewriter receives it."""
    return FakeClient([NS(type="tool_use", input={"query": text})], "tool_use")


def test_no_flags_no_history_does_not_call_claude() -> None:
    sq = search_query("q", NoCallClient(), rewrite_on=False, hyde_on=False)
    assert (sq.question, sq.text, sq.passage, sq.cost_usd) == ("q", "q", False, 0.0)


def test_follow_up_is_rewritten_with_history_and_answered_standalone() -> None:
    """A follow-up's pronoun only resolves through the earlier question, and
    `answer` takes no history, so Claude must get the standalone question."""
    client = _query("What were the reasons for ANDA refuse-to-receive decisions?")
    sq = search_query(
        "What were the main reasons for those?",
        client,
        ["What portion of ANDAs were refused in FY2018?"],
        rewrite_on=False,
        hyde_on=False,
    )
    prompt = client.calls[0]["messages"][0]["content"]
    assert "<earlier_question>What portion of ANDAs" in prompt
    assert sq.question == sq.text == client.response.content[0].input["query"]
    assert sq.cost_usd > 0


def test_rewrite_flag_changes_search_not_the_question_answered() -> None:
    sq = search_query("ANDAs FY12", _query("How many ANDAs in FY12?"), rewrite_on=True)
    assert (sq.question, sq.text) == ("ANDAs FY12", "How many ANDAs in FY12?")


def test_hyde_passage_embedded_without_query_prefix() -> None:
    """BGE is asymmetric: the HyDE passage goes on the passage side, like the
    indexed chunks; a question keeps QUERY_PREFIX."""
    sq = search_query("q", _reply("FDA approved 50 drugs."), hyde_on=True)
    assert sq.passage and sq.text == "FDA approved 50 drugs." and sq.question == "q"
    seen: list[str] = []
    r = _retriever(NS(query=lambda **kwargs: NS(matches=[])))
    r.model = NS(
        encode=lambda text, normalize_embeddings: seen.append(text)
        or NS(tolist=lambda: [0.0])
    )
    r.search(sq.text, passage=sq.passage)
    r.search("q")
    assert seen == ["FDA approved 50 drugs.", QUERY_PREFIX + "q"]


def test_tokenize_lowercases_and_splits_on_punctuation() -> None:
    """Chunks and queries go through the same function, so "COVID-19" in a query
    matches "covid-19" in a chunk."""
    assert tokenize("COVID-19 and Section 301(ll)") == [
        "covid",
        "19",
        "and",
        "section",
        "301",
        "ll",
    ]


def test_rrf_sums_ranks_across_lists_best_first() -> None:
    """b is 2nd in both lists and beats a, which is 1st in one list only;
    c and d tie (rank 3 in one list each) and are ordered by id."""
    fused = rrf([["a", "b", "d"], ["x", "b", "c"]], k=60)
    assert [cid for cid, _ in fused] == ["b", "a", "x", "c", "d"]
    assert fused[0][1] == pytest.approx(2 / 62)


def _bm25_retriever() -> Retriever:
    """5 chunks: "dscsa" in 2 (positive IDF), "the" in 3 (negative IDF)."""
    chunks = [
        Chunk("book-1-0", "book", 1, "The DSCSA product tracing rules.", 6, "text"),
        Chunk("deck-5-0", "deck", 5, "Bar chart of the DSCSA dates.", 6, "vision"),
        Chunk("deck-6-0", "deck", 6, "Opioid packaging and the docket.", 6, "text"),
        Chunk("deck-7-0", "deck", 7, "Biosimilar interchangeability.", 4, "text"),
    ]
    index = NS(query=lambda **kwargs: NS(matches=[NS(id="deck-4-0", score=0.9)]))
    return _retriever(index, "hybrid", chunks)


def test_bm25_query_matching_no_chunk_returns_nothing() -> None:
    """All-zero BM25 scores must not hand arbitrary chunks RRF credit."""
    r = _bm25_retriever()
    assert r.bm25_search("zzzz qqqq") == []
    assert [h.chunk.chunk_id for h in r.search("zzzz qqqq")] == ["deck-4-0"]


def test_words_in_most_chunks_add_no_bm25_score() -> None:
    """With rank_bm25's default epsilon, "the" (in 3 of 5 chunks) would get a
    positive IDF floor; BM25_EPSILON = 0 makes it worth nothing."""
    r = _bm25_retriever()
    assert r.bm25_search("the") == []
    assert r.bm25.idf["the"] == 0.0


def test_hybrid_fuses_vector_and_bm25_with_rrf_scores() -> None:
    """Vector finds deck-4-0 only; BM25 finds both DSCSA chunks (vision chunks
    are in the BM25 index too). Every hit carries its RRF score."""
    hits = _bm25_retriever().search("DSCSA")
    assert {h.chunk.chunk_id for h in hits} == {"deck-4-0", "book-1-0", "deck-5-0"}
    scores = {h.chunk.chunk_id: h.score for h in hits}
    assert scores["deck-4-0"] == pytest.approx(1 / 61)  # rank 1 in the vector list
    assert sorted(scores.values(), reverse=True) == [h.score for h in hits]


def test_every_citation_carries_its_source_metadata() -> None:
    """Whatever the retrieval scores (here negative cross-encoder logits), each
    citation names the chunk, PDF, 0-indexed page, score and generated flag of
    the document it quotes; page 0 included."""
    hits = [
        _hit(score=-1.5, page=0),
        _hit(score=-3.25, page=3, text="Chart: 50 approvals in 2021.", kind="vision"),
    ]
    content = [
        NS(type="text", text="50 drugs", citations=[_cite(0, 36, TEXT[:36])]),
        NS(type="text", text=" (chart).", citations=[_cite(7, 19, "50 approvals", 1)]),
    ]
    result = answer("How many?", hits, FakeClient(content))
    assert len(result.citations) == 2
    for c, h in zip(result.citations, hits):
        assert (c.chunk_id, c.doc_id, c.page) == (
            h.chunk.chunk_id,
            h.chunk.doc_id,
            h.chunk.page,
        )
        assert c.score == h.score and c.generated == (h.chunk.kind == "vision")
        assert c.cited_text and c.cited_text in h.chunk.text


class FakeCrossEncoder:
    """Scores a pair by how many of the question's words the chunk contains;
    records the pairs it was given."""

    def __init__(self) -> None:
        self.pairs: list[tuple[str, str]] = []

    def predict(self, pairs: list[tuple[str, str]]) -> list[float]:
        self.pairs += pairs
        return [float(sum(w in t.lower() for w in q.lower().split())) for q, t in pairs]


def _reranking_retriever(ce: FakeCrossEncoder) -> Retriever:
    """Vector order: deck-4-0, deck-6-0, deck-7-0 (cosine 0.9, 0.8, 0.7)."""
    chunks = [
        Chunk("deck-6-0", "deck", 6, "Opioid packaging and the docket.", 6, "text"),
        Chunk("deck-7-0", "deck", 7, "Opioid docket packaging workshop.", 4, "text"),
    ]
    ids = [("deck-4-0", 0.9), ("deck-6-0", 0.8), ("deck-7-0", 0.7)]
    matches = [NS(id=i, score=s) for i, s in ids]
    index = NS(query=lambda **kwargs: NS(matches=matches))
    return _retriever(index, "vector", chunks, ce)


def test_rerank_reorders_by_cross_encoder_score_and_keeps_every_hit() -> None:
    hits = _reranking_retriever(FakeCrossEncoder()).search("opioid docket workshop")
    assert [h.chunk.chunk_id for h in hits] == ["deck-7-0", "deck-6-0", "deck-4-0"]
    assert [h.score for h in hits] == [3.0, 2.0, 0.0]


def test_rerank_scores_the_question_not_the_hyde_passage() -> None:
    """Under HYDE the searched text is an invented passage; the cross-encoder
    must judge chunks against the question Claude answers."""
    ce = FakeCrossEncoder()
    _reranking_retriever(ce).search(
        "Invented FDA passage.", passage=True, question="opioid docket"
    )
    assert {q for q, _ in ce.pairs} == {"opioid docket"}


def test_rerank_of_no_hits_does_not_call_the_model() -> None:
    index = NS(query=lambda **kwargs: NS(matches=[]))
    assert _retriever(index, reranker=NoCallClient()).search("q") == []


def test_query_with_no_keyword_match_still_returns_ranked_hits() -> None:
    """Known bad query: punctuation only, so no BM25 token. Hybrid falls back to
    the vector list, the reranker still scores it, nothing raises."""
    ce = FakeCrossEncoder()
    base = _bm25_retriever()
    r = _retriever(base.index, "hybrid", list(base.chunks.values()), ce)
    hits = r.search("???")
    assert r.bm25_search("???") == []
    assert [h.chunk.chunk_id for h in hits] == ["deck-4-0"]
    assert ce.pairs == [("???", TEXT)]


# ------------------------------------------------------------ self-check (Day 34)

CALL_COST = (1000 * 0.10 + 100 * 0.50) / 1e6  # every fake call's usage


class SeqClient:
    """Returns canned Messages API responses in order; records the calls."""

    def __init__(self, *responses: NS) -> None:
        self.responses = list(responses)
        self.messages = self
        self.calls: list[dict] = []

    def create(self, **kwargs) -> NS:
        assert "temperature" not in kwargs
        self.calls.append(kwargs)
        return self.responses.pop(0)


def _resp(content: list, stop_reason: str = "end_turn") -> NS:
    return NS(
        content=content,
        stop_reason=stop_reason,
        usage=NS(input_tokens=1000, output_tokens=100),
    )


def _check(score: int, gap: str = "") -> NS:
    return _resp([NS(type="tool_use", input={"score": score, "gap": gap})], "tool_use")


def _cited_answer() -> NS:
    return _resp([NS(type="text", text="50.", citations=[_cite(0, 36, TEXT[:36])])])


class FakeSearch:
    """Retriever stand-in: records searches, returns one hit."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, str | None]] = []

    def search(self, text: str, question: str | None = None, **kwargs) -> list[Hit]:
        self.calls.append((text, question))
        return [_hit(page=9)]


FIRST = Answer("50 drugs.", [], False, False, 0.001, 1.0)


@pytest.mark.parametrize(
    "first",
    [
        Answer(ABSTAIN_TEXT, [], True, False, 0.001, 1.0),
        Answer(REFUSAL_TEXT, [], False, True, 0.001, 1.0),
    ],
)
def test_abstain_or_refusal_is_final_and_not_checked(first: Answer) -> None:
    final, _ = check_and_retry("q", first, [_hit()], FakeSearch(), NoCallClient())
    assert final == first


def test_score_at_threshold_keeps_the_first_answer_without_a_search() -> None:
    search = FakeSearch()
    final, hits = check_and_retry("q", FIRST, [_hit()], search, SeqClient(_check(6)))
    assert (final.text, final.confidence, final.retried) == ("50 drugs.", 6, False)
    assert final.cost_usd == pytest.approx(0.001 + CALL_COST)
    assert search.calls == [] and hits == [_hit()]


def test_low_score_searches_once_with_the_gap_and_answers_the_question() -> None:
    """The gap only changes the search: the cross-encoder and Claude get the
    user's question. The second answer is final; its check only fills
    `confidence`, even when it is low again."""
    client = SeqClient(
        _check(3, "FY16 refusal rate"),
        _resp(
            [NS(type="tool_use", input={"query": "ANDA refusal rate FY16"})], "tool_use"
        ),
        _cited_answer(),
        _check(2, "still missing"),
    )
    search = FakeSearch()
    final, hits = check_and_retry("How many?", FIRST, [_hit()], search, client)
    assert search.calls == [("ANDA refusal rate FY16", "How many?")]
    assert (
        "<missing>FY16 refusal rate</missing>"
        in client.calls[1]["messages"][0]["content"]
    )
    assert client.calls[2]["messages"][0]["content"][-1]["text"] == "How many?"
    assert final.retried and final.confidence == 2 and not final.abstained
    assert hits == [_hit(page=9)] and len(client.calls) == 4
    assert final.cost_usd == pytest.approx(0.001 + 4 * CALL_COST)


def test_retry_that_abstains_is_final_and_not_checked_again() -> None:
    client = SeqClient(
        _check(2, "missing"),
        _resp([NS(type="tool_use", input={"query": "q2"})], "tool_use"),
        _resp([NS(type="text", text=ABSTAIN_TEXT, citations=None)]),
    )
    final, _ = check_and_retry("q", FIRST, [_hit()], FakeSearch(), client)
    assert final.abstained and final.retried and final.confidence is None
    assert len(client.calls) == 3
