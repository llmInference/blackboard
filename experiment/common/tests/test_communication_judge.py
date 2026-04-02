from experiment.common.run_communication_judge import _extract_json_object


def test_extract_json_object_accepts_plain_json():
    payload = _extract_json_object('{"score": 1, "reason": "aligned"}')
    assert payload == {"score": 1, "reason": "aligned"}


def test_extract_json_object_accepts_think_and_fenced_json():
    payload = _extract_json_object(
        "<think>reasoning</think>\n```json\n{\"score\": 0, \"reason\": \"mismatch\"}\n```"
    )
    assert payload == {"score": 0, "reason": "mismatch"}


def test_extract_json_object_accepts_prefixed_text():
    payload = _extract_json_object(
        "My judgment is below:\n{\"score\": 1, \"reason\": \"consistent with sender intent\"}"
    )
    assert payload == {"score": 1, "reason": "consistent with sender intent"}
