"""
run_survey.py가 생성한 {survey_id}__ALL.json에서 Q3 가중치를 집계한다.

persona별 가중치 평균·표준편차 및 raw_persona_weights를 JSON/CSV로 저장한다.
입력 경로·출력 경로는 run_survey.py의 OUT_DIR 및 SURVEY_ID와 맞추는 것이 좋다.
"""

import json
import logging
import math
import os
from pathlib import Path
from collections import defaultdict

# run_survey.py와 동일한 기본 경로 사용
_BASE_DIR = Path(__file__).resolve().parent
_DEFAULT_SURVEY_PATH = _BASE_DIR / ".." / "survey" / "survey.json"
_DEFAULT_OUT_DIR = _BASE_DIR / ".." / ".." / "data" / "survey" / "responses"

def _resolve_survey_path() -> str:
    if os.getenv("SURVEY_JSON"):
        return os.environ["SURVEY_JSON"]
    p = _DEFAULT_SURVEY_PATH.resolve()
    if not p.is_file() and (p.parent / "survey.json").is_file():
        return str(p.parent / "survey.json")
    return str(p)

SURVEY_JSON = _resolve_survey_path()
OUT_DIR = Path(os.getenv("OUT_DIR", str(_DEFAULT_OUT_DIR.resolve())))
SURVEY_ID = os.getenv("SURVEY_ID", "cooling_fog_priority_v1")

IN_PATH = OUT_DIR / f"{SURVEY_ID}__ALL.json"
OUT_JSON = OUT_DIR / f"{SURVEY_ID}__weights_summary.json"
OUT_CSV = OUT_DIR / f"{SURVEY_ID}__weights_summary.csv"


def get_q3_weight_keys(survey_path: str | Path) -> list[str]:
    """
    survey.json에서 Q3(정량 가중치 배분) 문항의 items만 읽어 가중치 키 목록을 반환한다.
    설문 문항을 코드에 하드코딩하지 않고 JSON을 단일 소스로 사용하기 위함이다.
    """
    path = Path(survey_path)
    if not path.is_file():
        raise FileNotFoundError(f"Survey file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        survey = json.load(f)
    sections = survey.get("sections") or []
    inner = (sections[0].get("sections") or []) if sections else []
    for sec in inner:
        for item in sec.get("items", []):
            if item.get("id") == "Q3" and item.get("type") == "weight_allocation":
                return list(item.get("items", []))
    return []

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def mean(xs):
    return sum(xs) / len(xs) if xs else 0.0


def std(xs):
    if len(xs) <= 1:
        return 0.0
    m = mean(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))


def main():
    weight_keys = get_q3_weight_keys(SURVEY_JSON)
    if not weight_keys:
        raise ValueError(
            "Q3 (weight_allocation) items not found in survey. Check SURVEY_JSON path."
        )

    if not IN_PATH.is_file():
        raise FileNotFoundError(
            f"Survey results not found: {IN_PATH}. Run run_survey.py first."
        )
    with open(IN_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    # run_survey.py 출력: answers.Q3.weights 또는 최상위 Q3(flat) 형식 모두 지원
    def _get_weights(record: dict) -> dict | None:
        answers = record.get("answers") or {}
        q3 = answers.get("Q3") or record.get("Q3")
        if not isinstance(q3, dict):
            return None
        # answers.Q3.weights 형식이면 weights 사용, 아니면 Q3 자체가 가중치 dict
        w = q3.get("weights") if "weights" in q3 else q3
        return w if isinstance(w, dict) and w else None

    vals = defaultdict(list)
    for r in data:
        w = _get_weights(r)
        if not w:
            logger.warning("Missing Q3.weights for persona_id=%s", r.get("persona_id"))
            continue
        for k in weight_keys:
            if k in w:
                vals[k].append(int(w[k]))

    summary = {
        "survey_id": SURVEY_ID,
        "n_personas": len(data),
        "weights_mean": {k: mean(vals[k]) for k in weight_keys},
        "weights_std": {k: std(vals[k]) for k in weight_keys},
        "raw_persona_weights": {
            r["persona_id"]: w
            for r in data
            if (w := _get_weights(r))
        },
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    logger.info("Saved: %s", OUT_JSON)

    with open(OUT_CSV, "w", encoding="utf-8") as f:
        f.write("metric,mean,std\n")
        for k in weight_keys:
            f.write(f"{k},{summary['weights_mean'][k]:.4f},{summary['weights_std'][k]:.4f}\n")
    logger.info("Saved: %s", OUT_CSV)


if __name__ == "__main__":
    main()
