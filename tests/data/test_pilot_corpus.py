from kvr.data.pilot_corpus import PilotPrompt


def test_pilot_prompt_dataclass_has_required_fields():
    p = PilotPrompt(
        id="wikitable_0",
        source="wikitable",
        context="abc",
        question="q?",
        answer="a",
    )
    assert p.id == "wikitable_0"
    assert p.source in {"wikitable", "synthetic_json"}
