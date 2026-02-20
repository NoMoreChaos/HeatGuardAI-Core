"""
LLM 기반 전문가 설문조사 실행 엔진.

100m 격자 단위 쿨링포그 설치 우선순위 산정을 위한 가중치 도출 설문을
JSON으로 정의된 설문·페르소나를 바탕으로 OpenAI Responses API로 실행하고,
persona별 결과 및 통합 ALL.json을 생성한다. 생성된 JSON은 aggregate.py에서 사용한다.

사용: OPENAI_API_KEY, (선택) OPENAI_MODEL, (선택) SURVEY_JSON, PERSONAS_JSON, OUT_DIR
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

from openai import OpenAI


# -----------------------------------------------------------------------------
# 설정: 환경변수만 사용하여 재현성·보안을 유지
# -----------------------------------------------------------------------------
_BASE_DIR = Path(__file__).resolve().parent
_DEFAULT_SURVEY_PATH = _BASE_DIR / ".." / "survey" / "survey.json"
_DEFAULT_PERSONAS_PATH = _BASE_DIR / ".." / "survey" / "personas.json"
_DEFAULT_OUT_DIR = _BASE_DIR / ".." / ".." / "data" / "survey" / "responses"

SURVEY_JSON = os.getenv("SURVEY_JSON", str(_DEFAULT_SURVEY_PATH.resolve()))
PERSONAS_JSON = os.getenv("PERSONAS_JSON", str(_DEFAULT_PERSONAS_PATH.resolve()))
OUT_DIR = Path(os.getenv("OUT_DIR", str(_DEFAULT_OUT_DIR.resolve())))
MODEL = os.getenv("OPENAI_MODEL", "gpt-5-mini")
REASONING_EFFORT = "low"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# 입력 로드
# -----------------------------------------------------------------------------
def load_survey(path: str | Path) -> dict[str, Any]:
    """
    설문 정의 JSON을 로드한다.
    문항은 반드시 JSON에서만 로드되며, 코드 내 하드코딩을 피하기 위함이다.
    """
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"Survey file not found: {p}")
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def load_personas(path: str | Path) -> list[dict[str, Any]]:
    """
    전문가 페르소나 목록을 로드한다.
    각 페르소나별로 1회씩 설문을 실행하는 단위가 된다.
    """
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"Personas file not found: {p}")
    with open(p, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("personas.json must be a JSON array")
    return data


# -----------------------------------------------------------------------------
# 설문 내부 sections 추출 (중첩 구조 대응)
# -----------------------------------------------------------------------------
def _get_inner_sections(survey: dict[str, Any]) -> list[dict[str, Any]]:
    """survey.sections가 1단계 래핑된 경우를 풀어 실제 섹션 리스트를 반환한다."""
    sections = survey.get("sections") or []
    if not sections:
        return []
    first = sections[0]
    return first.get("sections") or []


def _format_item_for_prompt(item: dict[str, Any]) -> str:
    """단일 문항을 사람이 읽을 수 있는 텍스트로 변환한다."""
    lines = [f"[{item.get('id', '?')}] {item.get('prompt', '')}"]
    if "options" in item:
        lines.append("  선택지: " + ", ".join(item["options"]))
    if "groups" in item and "items" in item:
        lines.append(f"  그룹: {', '.join(item['groups'])}")
        lines.append("  항목: " + ", ".join(item["items"]))
    if "items" in item and "total" in item:
        lines.append("  항목(가중치 합 " + str(item["total"]) + "): " + ", ".join(item["items"]))
    if "rows" in item and "scale" in item:
        lines.append("  행: " + ", ".join(item["rows"]))
        lines.append("  척도: " + ", ".join(item["scale"]))
    if "scale" in item and "rows" not in item:
        lines.append("  척도: " + ", ".join(item["scale"]))
    return "\n".join(lines)


def _build_survey_content(survey: dict[str, Any]) -> str:
    """모든 section과 문항을 사람이 읽을 수 있는 텍스트 블록으로 만든다."""
    sections = _get_inner_sections(survey)
    parts = []
    for sec in sections:
        title = sec.get("title", "")
        parts.append(f"## {title}")
        for item in sec.get("items", []):
            parts.append(_format_item_for_prompt(item))
        parts.append("")
    return "\n".join(parts).strip()


# -----------------------------------------------------------------------------
# 프롬프트 구성 (순서 고정: Base → Persona → Survey → Output Constraint)
# -----------------------------------------------------------------------------
def build_prompt(survey: dict[str, Any], persona: dict[str, Any]) -> str:
    """
    LLM에 전달할 최종 프롬프트를 구성한다.
    순서: [Base Instructions] → [Persona Instructions] → [Survey Content] → [Output Constraint]
    """
    base_instructions = survey.get("base_instructions")
    if isinstance(base_instructions, list):
        base_text = "\n".join(base_instructions)
    else:
        base_text = str(base_instructions or "")

    biases = persona.get("decision_bias") or []
    bias_bullets = "\n".join(f"- {b}" for b in biases)
    persona_block = f"""
[Persona]
- 이름: {persona.get('name', '')}
- 역할: {persona.get('role_description', '')}
- 판단 성향:
{bias_bullets}
"""

    survey_block = _build_survey_content(survey)
    output_constraint = """
[출력 규칙]
반드시 JSON만 출력하세요. 설명 문장, 마크다운, 코드블록 사용 금지.
"""

    return f"""[Base Instructions]
{base_text}
{persona_block}
[Survey Content]
{survey_block}
{output_constraint}"""


# -----------------------------------------------------------------------------
# LLM 호출
# -----------------------------------------------------------------------------
def call_llm(prompt: str) -> str:
    """
    OpenAI Responses API를 호출하여 응답 텍스트를 반환한다.
    API Key는 환경변수 OPENAI_API_KEY로만 로드된다.
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY environment variable is not set")
    client = OpenAI(api_key=api_key)
    resp = client.responses.create(
        model=MODEL,
        reasoning={"effort": REASONING_EFFORT},
        input=prompt,
    )
    text = getattr(resp, "output_text", None)
    if not text:
        parts = []
        for item in getattr(resp, "output", []) or []:
            for c in getattr(item, "content", []) or []:
                if getattr(c, "type", "") == "output_text":
                    parts.append(getattr(c, "text", ""))
        text = "\n".join(parts).strip()
    return text or ""


# -----------------------------------------------------------------------------
# 응답 파싱 및 검증
# -----------------------------------------------------------------------------
def _extract_json_string(raw: str) -> str:
    """첫 '{'와 마지막 '}' 사이만 추출해 재파싱 가능한 문자열로 만든다."""
    start = raw.find("{")
    end = raw.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("No JSON object found in response.")
    return raw[start : end + 1]


def _validate_weights_sum(answers: dict[str, Any]) -> None:
    """Q3 weights 합계가 100이 아니면 ValueError를 발생시킨다."""
    q3 = answers.get("Q3")
    if not q3 or not isinstance(q3, dict):
        return
    weights = q3.get("weights")
    if not weights or not isinstance(weights, dict):
        return
    total = sum(int(v) for v in weights.values() if isinstance(v, (int, float)))
    if total != 100:
        raise ValueError(f"Q3 weights sum must be 100, got {total}")


def parse_response(
    raw_text: str,
    survey_id: str,
    persona_id: str,
) -> dict[str, Any]:
    """
    LLM 응답 문자열을 파싱하고, Q3 가중치 합계를 검증한 뒤 구조화된 dict를 반환한다.
    JSON 파싱 실패 시 첫 '{'~마지막 '}' 구간만 추출해 재시도한다.
    """
    try:
        obj = json.loads(raw_text.strip())
    except json.JSONDecodeError:
        obj = json.loads(_extract_json_string(raw_text))
    if not isinstance(obj, dict):
        raise ValueError("Parsed response is not a JSON object")
    obj.setdefault("survey_id", survey_id)
    obj.setdefault("persona_id", persona_id)
    answers = obj.get("answers")
    if isinstance(answers, dict):
        _validate_weights_sum(answers)
    return obj


# -----------------------------------------------------------------------------
# 결과 저장
# -----------------------------------------------------------------------------
def save_result(result: dict[str, Any], path: str | Path) -> None:
    """단일 설문 결과를 JSON 파일로 저장한다."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)


# -----------------------------------------------------------------------------
# 메인: persona별 실행 및 ALL.json 생성
# -----------------------------------------------------------------------------
def run_one_persona(
    survey: dict[str, Any],
    persona: dict[str, Any],
    survey_id: str,
    out_dir: Path,
) -> dict[str, Any] | None:
    """
    한 페르소나에 대해 설문 1회 실행 후 결과를 저장하고 결과 dict를 반환한다.
    실패 시 로그만 남기고 None을 반환한다.
    """
    pid = persona.get("id", "unknown")
    try:
        prompt = build_prompt(survey, persona)
        raw = call_llm(prompt)
        result = parse_response(raw, survey_id, pid)
        out_path = out_dir / f"{survey_id}__{pid}.json"
        save_result(result, out_path)
        logger.info("Saved persona result: %s", out_path)
        return result
    except Exception as e:
        logger.exception("Persona %s failed: %s", pid, e)
        return None


def main() -> None:
    """모든 페르소나에 대해 설문을 실행하고, 개별 JSON과 ALL.json을 저장한다."""
    survey = load_survey(SURVEY_JSON)
    personas = load_personas(PERSONAS_JSON)
    survey_id = survey.get("survey_id", "survey")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    results: list[dict[str, Any]] = []
    for persona in personas:
        pid = persona.get("id", "?")
        result = run_one_persona(survey, persona, survey_id, OUT_DIR)
        if result is not None:
            results.append(result)
        else:
            logger.warning("Skipped (failed) persona_id: %s", pid)

    all_path = OUT_DIR / f"{survey_id}__ALL.json"
    save_result(results, all_path)
    logger.info("Saved combined results: %s (n=%d)", all_path, len(results))


if __name__ == "__main__":
    main()
