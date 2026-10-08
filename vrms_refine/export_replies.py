"""Export cached public assistant replies; never call either model API."""
import csv
import hashlib
import json
from pathlib import Path


OUT = Path("outputs/vrms_refine/20261008_seed2026")
MODELS = {"gpt": "gpt-6.1-sol", "deepseek": "deepseek-flash"}


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def public_reply(log, vendor):
    if not log["success"]:
        raise ValueError("A failed call cannot be displayed as an agent reply")
    prediction = log["predictions"][0] if vendor == "gpt" else log["prediction"]
    response = next(a["response"] for a in reversed(log["attempts"]) if a.get("status") == "success")
    if response["model"] != MODELS[vendor]:
        raise ValueError("Unexpected returned model")
    if vendor == "gpt":
        reply = "".join(part["text"] for item in response.get("output", [])
                        for part in item.get("content", []) if part.get("type") == "output_text")
    else:
        reply = response["choices"][0]["message"]["content"]
    if not isinstance(reply, str) or not reply.strip():
        raise ValueError("The public final assistant message is missing")
    return prediction, reply


def main():
    root = Path.cwd().resolve()
    records = []
    for vendor in MODELS:
        baseline = Path("outputs/vrms_cloud/20261008_seed2026/independent_calls") if vendor == "gpt" else Path("outputs/vrms_deepseek/20261008_seed2026/independent_calls")
        with (OUT / vendor / "path_oof.csv").open(encoding="utf-8-sig", newline="") as f:
            rows = list(csv.DictReader(f))
        fresh = {r["path_index"]: r for r in read_json(OUT / vendor / "run_status.json")["results"] if r["role"] == "outer_test"}
        if len(rows) != 147 or len(fresh) != 147:
            raise ValueError("Incomplete outer-path reply set")
        for row in rows:
            i = int(row["path_index"])
            enhanced = Path(fresh[i].get("selected_log_path", str(OUT / vendor / "calls/enhanced" / f"fold_{fresh[i]['outer_subject']:02d}_path_{i:03d}.json")))
            for stage, path, score in (("baseline", baseline / f"path_{i:03d}_cloud_call.json", "baseline_cloud"),
                                       ("enhanced", enhanced, "enhanced_cloud")):
                path = path.resolve()
                if not path.is_relative_to(root):
                    raise ValueError("Reply source is outside the workspace")
                raw = path.read_bytes()
                prediction, message = public_reply(json.loads(raw), vendor)
                if prediction["high_probability"] != float(row[score]):
                    raise ValueError("The displayed reply differs from the evaluated score")
                records.append(dict(path_index=i, vendor=vendor, model=MODELS[vendor], stage=stage,
                                    prediction=prediction, assistant_message=message,
                                    source_path=path.as_posix(), source_sha256=hashlib.sha256(raw).hexdigest()))
    if len(records) != 588:
        raise ValueError("Expected 147 paths, two models, and two prompt versions")
    index = {(r["path_index"], r["vendor"], r["stage"]): r for r in records}
    lines = ["# Agent真实回复：GPT与DeepSeek改善前后对照", "",
             "这里展示147条外层测试路径、两家模型、两种提示版本的588条已保存回复。本次只读取缓存，没有新增模型API调用。", "",
             "代码块保留API返回的最终助手消息原文；高低状态、概率、不确定标志和英文理由均未改写。当前接口要求结构化JSON和短理由，未生成长篇聊天报告。", "",
             "67.35%的GPT方案属于原25%融合后的本地概率校准，校准没有产生新的云端回复。下面的高类概率是LLM原始输出，融合及校准结果见各模型的path_oof.csv。", "",
             "路径编号用于本地索引；原请求中的匿名查询编号为qsingle。", "",
             "[GPT融合与校准数值](" + (root / OUT / "gpt/path_oof.csv").as_posix() + ") · [DeepSeek融合与校准数值](" + (root / OUT / "deepseek/path_oof.csv").as_posix() + ")", "",
             "可以用文件搜索定位“路径004”等编号。", ""]
    for i in range(147):
        lines += [f"## 路径{i:03d}", "",
                  "| 模型 | 提示 | 判断 | 高类概率 | 自报不确定 |",
                  "|---|---|---|---:|---|"]
        for vendor in MODELS:
            for stage, label in (("baseline", "原提示"), ("enhanced", "增强提示")):
                p = index[i, vendor, stage]["prediction"]
                lines.append(f"| {MODELS[vendor]} | {label} | {'高眩晕' if p['state'] == 'high' else '低眩晕'} | {p['high_probability']:.0%} | {'是' if p['uncertain'] else '否'} |")
        lines.append("")
        for vendor in MODELS:
            lines += [f"### {MODELS[vendor]}", ""]
            for stage, label in (("baseline", "原提示"), ("enhanced", "增强提示")):
                record = index[i, vendor, stage]
                fence = "`" * max(4, max((len(s) for s in record["assistant_message"].splitlines() if s and set(s) == {"`"}), default=0) + 1)
                lines += [f"**{label}：最终助手消息原文**", "", fence + "text", record["assistant_message"], fence, "",
                          f"[对应调用记录]({record['source_path']})", ""]
    (OUT / "Agent真实回复_改善前后.md").write_text("\n".join(lines), encoding="utf-8")
    (OUT / "agent_replies.json").write_text(json.dumps(dict(reply_count=len(records), new_model_api_calls=0, records=records), ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(dict(status="exported", paths=147, replies=len(records), new_model_api_calls=0), ensure_ascii=False))


if __name__ == "__main__":
    main()
