# -*- coding: utf-8 -*-
"""Question coverage over synthetic task/rest EEG tool JSON, without baselines.

No subject answer, questionnaire, provider, model, or training job is accessed.
"""
import copy

import pytest

from eeg_agent.brain import BrainSession, _numbers, _tool_facts
from eeg_agent.recordings import recording_tool
from eeg_agent.synthesis import question_scopes, requirements, validate_answer


QUESTION = (
    "请三个 Agent 协作回答：\n"
    "2. 疲劳相关 EEG 指标第一条与最后一条完整路径相比，是上升还是下降？\n"
    "3. 心理负荷 EEG 指标哪条完整路径的指标最高？"
)
REST_QUESTION = (
    "1. 疲劳相关 EEG 指标第一段与最后一段完整休息相比，是上升还是下降？\n"
    "2. 心理负荷 EEG 指标哪段完整休息的指标最高？"
)
BOTH_QUESTION = QUESTION + "\n" + REST_QUESTION
FATIGUE_TEXT = (
    "2. 疲劳指标（路径）：完整路径汇总指标2.0000。第一条完整路径为第1条"
    "（01:00.000—02:00.000），指标3.2000；最后一条完整路径为第3条"
    "（06:00.000—07:00.000），指标1.9200，首末比较下降40.00%。来源：fatigue.raw_spectrum。"
)
WORKLOAD_TEXT = (
    "3. 心理负荷（路径）：完整路径汇总指标2.8000。最高的完整路径为第1条"
    "（01:00.000—02:00.000），指标7.2000。来源：emotion.raw_workload。"
)
FATIGUE_REST_TEXT = (
    "1. 疲劳指标（休息）：完整休息段汇总指标4.4000。第一段完整休息为第1段"
    "（02:00.000—03:00.000），指标8.0000；最后一段完整休息为第3段"
    "（07:00.000—08:00.000），指标4.0000，首末比较下降50.00%。来源：fatigue.raw_spectrum。"
)
WORKLOAD_REST_TEXT = (
    "2. 心理负荷（休息）：完整休息段汇总指标3.9000。最高的完整休息为第1段"
    "（02:00.000—03:00.000），指标6.0000。来源：emotion.raw_workload。"
)


def change(text):
    return {"text": text, "direction": text[:2]}


def state_facts(first, last, overall, state, change_text):
    intervals = [("01:00.000", "02:00.000"), ("03:00.000", "04:00.000"),
                 ("06:00.000", "07:00.000"), ("08:00.000", "08:30.000")]
    if state == "rest":
        intervals = [("02:00.000", "03:00.000"), ("04:00.000", "05:00.000"),
                     ("07:00.000", "08:00.000"), ("09:00.000", "09:30.000")]
    rows = [{"ordinal": ordinal, "complete": ordinal < 4, "value": value,
             "start_clock": interval[0], "end_clock": interval[1]}
            for ordinal, (value, interval) in enumerate(
                zip((first, (first + last) / 2, last, 99.0), intervals), start=1)]
    a, b = rows[0], rows[2]
    return {"value": overall, "segments": rows,
            "first_to_last": {
                "first_ordinal": 1, "last_ordinal": 3, "first_value": first, "last_value": last,
                "first_start_clock": a["start_clock"], "first_end_clock": a["end_clock"],
                "last_start_clock": b["start_clock"], "last_end_clock": b["end_clock"],
                "change": change(change_text)},
            "max_complete": {"value": first, "segments": [copy.deepcopy(a)],
                             "compared_count": 3, "missing_count": 0}}


def synthetic_tool(source, task, rest):
    measured = {"segment_scope": "task_paths", "task_value": task["value"],
                "segments": task["segments"], "first_to_last": task["first_to_last"],
                "max_complete": task["max_complete"]}
    measured.update({"rest_" + key: value for key, value in rest.items()})
    return {"tool": source, "result": {"available": True,
            "reason": "已从原始 EEG 按路径与休息状态计算指标", "measurements": measured}}


@pytest.fixture
def tools():
    return [synthetic_tool("fatigue.raw_spectrum",
                          state_facts(3.2, 1.92, 2.0, "task", "下降40.00%"),
                          state_facts(8.0, 4.0, 4.4, "rest", "下降50.00%")),
            synthetic_tool("emotion.raw_workload",
                          state_facts(7.2, 3.6, 2.8, "task", "下降50.00%"),
                          state_facts(6.0, 3.0, 3.9, "rest", "下降50.00%"))]


def reply(required, text):
    facts = {(fact["source"], fact["path"]): fact
             for item in required for fact in item["required_facts"]}
    return {"explanation": text, "citations": [], "measurement_claims": list(facts.values())}


def check(tools, text, question=QUESTION):
    required = requirements(question, tools)
    return validate_answer(reply(required, text), required, _numbers)


def test_numbered_q2_q3_keep_domains_and_path_questions_separate(tools):
    scopes = question_scopes(QUESTION)
    assert "最后一条" in scopes["fatigue"] and "最高" not in scopes["fatigue"]
    assert "最高" in scopes["emotion"] and "最后一条" not in scopes["emotion"]
    required = requirements(QUESTION, tools)
    assert {item["id"] for item in required} == {
        "fatigue.measurement", "fatigue.first_last", "emotion.measurement", "emotion.maximum"}
    assert {item["state"] for item in required} == {"task"}
    assert not any(item["source"].startswith("vrms.") for item in required)


def test_natural_unnumbered_questions_keep_domain_and_state_assignments(tools):
    question = "疲劳首末完整路径如何变化？心理负荷休息段最高是哪段？"
    required = requirements(question, tools)
    assert {item["id"] for item in required} == {
        "fatigue.measurement", "fatigue.first_last", "emotion.rest_measurement", "emotion.rest_maximum"}


def test_complete_path_answer_resolves_endpoint_and_peak_with_exact_intervals(tools):
    review = check(tools, FATIGUE_TEXT + "\n\n" + WORKLOAD_TEXT)
    assert review["verified_items"] == review["requested_items"] == 4
    assert review["topics"] == ["疲劳", "心理负荷"]


def test_rest_endpoint_ordinals_accept_natural_ge_counter_without_losing_binding(tools):
    text = FATIGUE_REST_TEXT.replace('第1段', '第1个完整休息段').replace('第3段', '第3个')
    text += '\n\n' + WORKLOAD_REST_TEXT
    assert check(tools, text, REST_QUESTION)['verified_items'] == 4
    swapped = text.replace('指标8.0000', '指标4.0000').replace('指标4.0000，首末', '指标8.0000，首末')
    with pytest.raises(ValueError):
        check(tools, swapped, REST_QUESTION)


def test_rest_only_requirements_use_rest_facts_without_task_or_baseline_fields(tools):
    required = requirements(REST_QUESTION, tools)
    assert {item["id"] for item in required} == {
        "fatigue.rest_measurement", "fatigue.rest_first_last",
        "emotion.rest_measurement", "emotion.rest_maximum"}
    assert {item["state"] for item in required} == {"rest"}
    assert all(fact["path"].startswith("measurements.rest_")
               for item in required for fact in item["required_facts"])
    assert check(tools, FATIGUE_REST_TEXT + "\n\n" + WORKLOAD_REST_TEXT, REST_QUESTION)["verified_items"] == 4


def test_rest_request_with_generic_format_instruction_does_not_add_task_need(tools):
    question = "疲劳休息段的第一段与最后一段相比如何变化？请给结论、数值和时间区间。"
    assert {item["state"] for item in requirements(question, tools)} == {"rest"}
    assert check(tools, FATIGUE_REST_TEXT, question)["verified_items"] == 2


def test_rest_followup_clause_inherits_state_without_repeating_the_noun(tools):
    required = requirements("疲劳完整休息段哪个最高？首末是否上升？", tools)
    assert {item["id"] for item in required} == {
        "fatigue.rest_measurement", "fatigue.rest_maximum", "fatigue.rest_first_last"}
    assert {item["state"] for item in required} == {"rest"}


def test_four_domain_state_answers_remain_independently_covered(tools):
    text = "\n\n".join((FATIGUE_TEXT, WORKLOAD_TEXT, FATIGUE_REST_TEXT, WORKLOAD_REST_TEXT))
    assert check(tools, text, BOTH_QUESTION)["verified_items"] == 8


def test_task_answer_cannot_supply_missing_rest_answer_with_all_claims(tools):
    with pytest.raises(ValueError):
        check(tools, FATIGUE_TEXT + "\n\n" + WORKLOAD_TEXT, BOTH_QUESTION)


def test_path_one_and_rest_one_cannot_swap_time_and_value_bindings(tools):
    text = FATIGUE_REST_TEXT.replace("（02:00.000—03:00.000），指标8.0000",
                                     "（01:00.000—02:00.000），指标3.2000")
    with pytest.raises(ValueError):
        check(tools, text + "\n\n" + WORKLOAD_REST_TEXT, REST_QUESTION)


def test_task_values_cannot_replace_rest_state_overall_values(tools):
    text = FATIGUE_REST_TEXT.replace("汇总指标4.4000", "汇总指标2.0000")
    with pytest.raises(ValueError):
        check(tools, text + "\n\n" + WORKLOAD_REST_TEXT, REST_QUESTION)


@pytest.mark.parametrize("remove", ["（07:00.000—08:00.000）", "指标4.0000，", "指标6.0000。"])
def test_rest_endpoint_peak_or_interval_must_be_visible_not_only_claimed(tools, remove):
    text = (FATIGUE_REST_TEXT + "\n\n" + WORKLOAD_REST_TEXT).replace(remove, "")
    with pytest.raises(ValueError):
        check(tools, text, REST_QUESTION)


def test_same_change_percent_in_two_states_cannot_hide_rest_direction_flip(tools):
    measured = tools[0]["result"]["measurements"]
    measured["first_to_last"]["last_value"] = 1.6
    measured["first_to_last"]["change"] = change("下降50.00%")
    task = FATIGUE_TEXT.replace("指标1.9200", "指标1.6000").replace("下降40.00%", "下降50.00%")
    rest = FATIGUE_REST_TEXT.replace("下降50.00%", "上升50.00%")
    with pytest.raises(ValueError):
        check(tools, "\n\n".join((task, WORKLOAD_TEXT, rest, WORKLOAD_REST_TEXT)), BOTH_QUESTION)


def test_record_start_cannot_create_baseline_obligations(tools):
    required = requirements("疲劳指标相对任务前基线怎样变化？第一与最后完整路径相比如何？", tools)
    assert not any("baseline" in item["id"] or any("baseline" in f["path"]
                  for f in item["required_facts"]) for item in required)


def test_natural_path_wording_is_accepted_when_comparisons_are_complete(tools):
    text = FATIGUE_TEXT.replace("疲劳指标（路径）：", "疲劳相关 EEG 指标在路径间下降：")
    text += "\n\n" + WORKLOAD_TEXT.replace("心理负荷（路径）：", "心理负荷 EEG 指标在路径中：")
    assert check(tools, text)["verified_items"] == 4


def test_integrated_conclusion_cannot_replace_missing_detailed_state_values(tools):
    conclusion = "综合结论：疲劳路径汇总指标2.0000，心理负荷路径汇总指标2.8000。"
    text = conclusion + "\n\n" + FATIGUE_TEXT.replace("完整路径汇总指标2.0000。", "")
    with pytest.raises(ValueError):
        check(tools, text + "\n\n" + WORKLOAD_TEXT)


@pytest.mark.parametrize("text", [
    "疲劳指标（路径）：汇总指标2.0000。来源：fatigue.raw_spectrum。\n\n" + WORKLOAD_TEXT,
    FATIGUE_TEXT + "\n\n心理负荷（路径）：汇总指标2.8000。来源：emotion.raw_workload。",
    FATIGUE_TEXT.replace("（06:00.000—07:00.000）", "") + "\n\n" + WORKLOAD_TEXT,
    FATIGUE_TEXT + "\n\n" + WORKLOAD_TEXT.replace("指标7.2000。", ""),
])
def test_all_exact_claims_cannot_hide_missing_path_comparison_or_interval(tools, text):
    with pytest.raises(ValueError):
        check(tools, text)


def test_missing_endpoint_claim_is_rejected_even_with_complete_prose(tools):
    required = requirements(QUESTION, tools)
    value = reply(required, FATIGUE_TEXT + "\n\n" + WORKLOAD_TEXT)
    value["measurement_claims"] = [c for c in value["measurement_claims"]
                                   if c["path"] != "measurements.first_to_last.last_value"]
    with pytest.raises(ValueError):
        validate_answer(value, required, _numbers)


def test_correct_numbers_do_not_validate_reversed_path_change(tools):
    with pytest.raises(ValueError):
        check(tools, FATIGUE_TEXT.replace("下降40.00%", "上升40.00%") + "\n\n" + WORKLOAD_TEXT)


def test_old_agent_prefixed_repetition_is_not_a_supervisor_answer(tools):
    with pytest.raises(ValueError, match="repeated specialist"):
        check(tools, "FatigueAgent：" + FATIGUE_TEXT + "\n\nEmotionAgent：" + WORKLOAD_TEXT)


@pytest.mark.parametrize("separator", ["\n\n", " "])
def test_two_domains_cannot_swap_overall_values(tools, separator):
    text = FATIGUE_TEXT.replace("汇总指标2.0000", "汇总指标2.8000")
    text += separator + WORKLOAD_TEXT.replace("汇总指标2.8000", "汇总指标2.0000")
    with pytest.raises(ValueError):
        check(tools, text)


def test_endpoint_values_cannot_be_attached_to_wrong_path(tools):
    text = FATIGUE_TEXT.replace("指标3.2000", "指标1.9200").replace(
        "指标1.9200，首末", "指标3.2000，首末")
    with pytest.raises(ValueError):
        check(tools, text + "\n\n" + WORKLOAD_TEXT)


def test_endpoint_intervals_cannot_be_attached_to_wrong_path(tools):
    text = FATIGUE_TEXT.replace("01:00.000—02:00.000", "INTERVAL")
    text = text.replace("06:00.000—07:00.000", "01:00.000—02:00.000")
    text = text.replace("INTERVAL", "06:00.000—07:00.000")
    with pytest.raises(ValueError):
        check(tools, text + "\n\n" + WORKLOAD_TEXT)


@pytest.mark.parametrize("state", ["task", "rest"])
def test_tool_source_must_be_visible_in_each_domain_state_paragraph(tools, state):
    text = (FATIGUE_TEXT + "\n\n" + WORKLOAD_TEXT if state == "task" else
            FATIGUE_REST_TEXT + "\n\n" + WORKLOAD_REST_TEXT).replace("来源：fatigue.raw_spectrum。", "")
    with pytest.raises(ValueError):
        check(tools, text, QUESTION if state == "task" else REST_QUESTION)


def test_topic_cannot_attribute_its_measurement_to_another_domains_tool(tools):
    with pytest.raises(ValueError):
        check(tools, FATIGUE_TEXT.replace("fatigue.raw_spectrum", "emotion.raw_workload") + "\n\n" + WORKLOAD_TEXT)


def test_chinese_tool_names_can_attribute_measurements(tools):
    text = FATIGUE_TEXT.replace("fatigue.raw_spectrum", "原始频谱工具")
    text += "\n\n" + WORKLOAD_TEXT.replace("emotion.raw_workload", "原始 EEG 工作负荷工具")
    assert check(tools, text)["verified_items"] == 4


def test_maximum_cannot_be_reported_as_the_minimum(tools):
    with pytest.raises(ValueError):
        check(tools, FATIGUE_TEXT + "\n\n" + WORKLOAD_TEXT.replace("最高的完整路径", "最低的完整路径"))


def test_unrequested_tools_do_not_add_domain_or_state_needs(tools):
    required = requirements("心理负荷哪条完整路径最高？", tools)
    assert {item["id"] for item in required} == {"emotion.measurement", "emotion.maximum"}
    assert validate_answer(reply(required, WORKLOAD_TEXT), required, _numbers)["verified_items"] == 2


def test_prompt_carries_state_needs_without_record_start_baseline(tools):
    class CaptureProvider:
        def call(self, role, payload, messages):
            self.captured = (role, copy.deepcopy(payload), copy.deepcopy(messages))
            return {"explanation": "not rendered in this prompt-contract test"}
    session = object.__new__(BrainSession)
    session.recording_id, session.provider = "synthetic-recording", CaptureProvider()
    required = requirements(BOTH_QUESTION, tools)
    session._call("Supervisor", {"question": BOTH_QUESTION, "agent_reports": [],
                  "answer_requirements": required, "tool_facts": _tool_facts(tools)},
                  "The final VRMS class must come only from measurements.cloud_judgment.final_class.")
    role, payload, messages = session.provider.captured
    instruction = messages[0]["content"]
    assert role == "Supervisor" and payload["generation_phase"] == "supervisor_report"
    assert payload["answer_requirements"] == required
    assert {item["state"] for item in required} == {"task", "rest"}
    assert "measurements.cloud_judgment.final_class" not in instruction
    assert "Baseline-relative change" not in instruction
    assert "Copy percentage descriptions from baseline_change.text" not in instruction
    assert "not a transcript" in instruction and "never by agent names" in instruction
    assert "questionnaire answers" in instruction and "No identity is needed" in instruction


@pytest.mark.parametrize("omit_reason", [False, True])
def test_no_events_gives_specific_endpoint_and_ranking_reason(tools, omit_reason):
    fatigue, workload = (tool["result"]["measurements"] for tool in tools)
    fatigue.update(first_to_last=None, first_to_last_unavailable_reason="原始记录没有任务事件，首末完整路径比较未定义")
    workload.update(max_complete=None, max_complete_unavailable_reason="原始记录没有任务事件，完整路径排名未定义")
    text = "疲劳指标（路径）：汇总指标2.0000。原始记录没有任务事件，首末完整路径比较未定义。来源：fatigue.raw_spectrum。"
    text += "\n\n心理负荷（路径）：汇总指标2.8000。"
    text += "" if omit_reason else "原始记录没有任务事件，完整路径排名未定义。"
    text += "来源：emotion.raw_workload。"
    if omit_reason:
        with pytest.raises(ValueError):
            check(tools, text)
    else:
        assert check(tools, text)["verified_items"] == 4


def test_missing_rest_value_cannot_be_reported_as_analysis_completed(tools):
    tools[0]["result"]["measurements"].update(rest_value=None, rest_segments=[], rest_first_to_last=None,
                    rest_max_complete=None, rest_unavailable_reason="原始记录没有完整休息事件段，休息指标未定义",
                    rest_first_to_last_unavailable_reason="没有取得有效指标的完整休息段，首末比较未定义",
                    rest_max_complete_unavailable_reason="没有取得有效指标的完整休息段，无法计算最高指标")
    with pytest.raises(ValueError):
        check(tools, "疲劳指标（休息）：分析完成。来源：fatigue.raw_spectrum。", "疲劳休息段的指标如何？")


def test_absent_rest_state_gives_its_own_reason_while_path_measurement_remains_available(tools):
    reason = "原始记录没有完整休息事件段，休息指标未定义"
    tools[0]["result"]["measurements"].update(rest_value=None, rest_segments=[], rest_unavailable_reason=reason)
    required = requirements("疲劳休息段的指标如何？", tools)
    assert len(required) == 1 and required[0]["state"] == "rest"
    assert required[0]["required_facts"] == [{"source": "fatigue.raw_spectrum",
            "path": "measurements.rest_unavailable_reason", "value": reason}]
    text = "疲劳指标（休息）：" + reason + "。来源：fatigue.raw_spectrum。"
    assert validate_answer(reply(required, text), required, _numbers)["verified_items"] == 1


def test_unavailable_tool_cannot_hide_behind_analysis_completed(tools):
    tools[0]["result"].update(available=False, reason="没有通过质量检查的完整五秒 EEG 窗口")
    with pytest.raises(ValueError):
        check(tools, "疲劳指标（路径）：分析完成。来源：fatigue.raw_spectrum。\n\n" + WORKLOAD_TEXT)


def raw_evidence(values, complete=None, scope="task_paths", rest_values=(), rest_complete=None):
    def segments(items, finished, rest=False):
        return [{"ordinal": i + 1, "complete": done,
                 "start_clock": f"{i + 1:02d}:{'30' if rest else '00'}.000",
                 "end_clock": f"{i + 2:02d}:00.000" if rest else f"{i + 1:02d}:30.000",
                 "stats": {"theta_alpha": value, "workload_index": value,
                           "accepted_windows": 6 if value is not None else 0}}
                for i, (value, done) in enumerate(zip(items, finished))]
    return {"recording": {"segment_scope": scope, "signal_sha256": "synthetic-waveform",
                          "segments": segments(values, complete or [True] * len(values)),
                          "rest_segments": segments(rest_values, rest_complete or [True] * len(rest_values), True),
                          "overall": {"theta_alpha": 2.0, "workload_index": 2.8},
                          "rest_overall": {"theta_alpha": 4.4, "workload_index": 3.9}}}


def test_incomplete_path_is_excluded_from_peak_and_last_endpoint():
    measured = recording_tool("emotion.raw_workload", raw_evidence([7.0, 3.0, 99.0], [True, True, False]))["measurements"]
    assert measured["max_complete"]["value"] == 7.0
    assert [row["ordinal"] for row in measured["max_complete"]["segments"]] == [1]
    assert measured["first_to_last"]["last_ordinal"] == 2


def test_incomplete_rest_is_excluded_and_path_rows_are_not_substituted():
    evidence = raw_evidence([1.0, 2.0], rest_values=[9.0, 4.5, 99.0], rest_complete=[True, True, False])
    measured = recording_tool("fatigue.raw_spectrum", evidence)["measurements"]
    assert measured["rest_max_complete"]["value"] == 9.0
    assert measured["rest_first_to_last"]["first_value"] == 9.0
    assert measured["rest_first_to_last"]["last_ordinal"] == 2
    assert measured["first_to_last"]["first_value"] == 1.0


@pytest.mark.parametrize("state", ["task", "rest"])
def test_tied_peaks_require_every_state_specific_interval(state):
    tool = recording_tool("emotion.raw_workload", raw_evidence([7.0, 3.0, 7.0], rest_values=[8.0, 4.0, 8.0]))
    question = "心理负荷最高的完整路径有哪些？" if state == "task" else "心理负荷最高的完整休息段有哪些？"
    required = requirements(question, [{"tool": tool["tool"], "result": tool}])
    if state == "task":
        text = "心理负荷（路径）：汇总指标2.8000，最高指标7.0000，并列为第1条（01:00.000—01:30.000）和第3条（03:00.000—03:30.000）。来源：emotion.raw_workload。"
        missing = "和第3条（03:00.000—03:30.000）"
    else:
        text = "心理负荷（休息）：汇总指标3.9000，最高指标8.0000，并列为第1段（01:30.000—02:00.000）和第3段（03:30.000—04:00.000）。来源：emotion.raw_workload。"
        missing = "和第3段（03:30.000—04:00.000）"
    assert validate_answer(reply(required, text), required, _numbers)["verified_items"] == 2
    with pytest.raises(ValueError):
        validate_answer(reply(required, text.replace(missing, "")), required, _numbers)


def test_missing_true_first_task_or_rest_endpoint_is_not_silently_replaced():
    measured = recording_tool("fatigue.raw_spectrum", raw_evidence([None, 3.0, 2.0], rest_values=[None, 8.0, 4.0]))["measurements"]
    for prefix in ("", "rest_"):
        assert measured[prefix + "first_to_last"] is None
        assert "第 1 条" in measured[prefix + "first_to_last_unavailable_reason"]
        assert "不能替换首末端点" in measured[prefix + "first_to_last_unavailable_reason"]
        assert measured[prefix + "max_complete"]["segments"][0]["ordinal"] == 2
        assert measured[prefix + "max_complete"]["missing_count"] == 1


def test_no_event_record_never_becomes_a_task_or_rest_segment():
    measured = recording_tool("emotion.raw_workload", raw_evidence([2.8], scope="whole_recording"))["measurements"]
    for prefix in ("", "rest_"):
        assert measured[prefix + "first_to_last"] is None and measured[prefix + "max_complete"] is None
        assert "没有任务事件" in measured[prefix + "first_to_last_unavailable_reason"]
        assert "没有任务事件" in measured[prefix + "max_complete_unavailable_reason"]


def test_measurement_tool_has_no_record_start_baseline_values():
    measured = recording_tool("fatigue.raw_spectrum", raw_evidence([3.2, 1.92], rest_values=[8.0, 4.0]))["measurements"]
    assert measured.get("baseline_available") is False
    assert not {"baseline_value", "baseline_change", "baseline_interval"}.intersection(measured)
    assert "不使用记录开头" in measured["baseline_unavailable_reason"]
