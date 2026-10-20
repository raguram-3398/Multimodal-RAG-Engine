"""The app page without network calls: page load and the no-key path."""

from pathlib import Path

from streamlit.testing.v1 import AppTest

from multimodal_rag.config import EXAMPLE_QUESTIONS

APP = str(Path(__file__).resolve().parents[1] / "streamlit_app.py")


def test_page_loads_with_the_examples() -> None:
    at = AppTest.from_file(APP).run()
    assert not at.exception
    assert [b.label for b in at.sidebar.button] == list(EXAMPLE_QUESTIONS)


def test_question_without_a_key_runs_nothing() -> None:
    """No key: no search, no Claude call, and the question is not counted."""
    at = AppTest.from_file(APP).run()
    at.chat_input[0].set_value("ANDAs received by FDA in FY12").run()
    assert [w.value for w in at.warning] == [
        "Enter your Anthropic API key in the sidebar first."
    ]
    assert at.session_state["n_queries"] == 0 and at.session_state["turns"] == []
