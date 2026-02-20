"""
heatguard_ranking_data_cluster_1.csv에 쿨링포그 설치 추천 이유 LLM 요약(LLM_summary) 컬럼을 추가하는 스크립트.

- 입력: --csv (기본 heatguard_ranking_data_cluster_1.csv)
- 출력: --output / -o 로 지정. 미지정 시 입력과 같은 폴더에 입력파일명_llm_summary.csv 로 저장 (원본 덮어쓰지 않음).
- 전체 행: --all / 테스트 N건: --limit N (기본 3건)

사용: OPENAI_API_KEY 필수. OPENAI_MODEL 선택(기본 gpt-4.1).
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
from pathlib import Path

def _debug(msg: str) -> None:
    """디버깅 메시지 (즉시 플러시)."""
    print(f"[DEBUG] {msg}", flush=True)


try:
    import pandas as pd
    from openai import OpenAI
except ImportError as e:
    print(f"필요한 패키지가 없습니다: {e}", file=sys.stderr, flush=True)
    print("다음으로 설치 후 다시 실행하세요: pip install pandas openai", file=sys.stderr, flush=True)
    sys.exit(1)

# -----------------------------------------------------------------------------
# 경로·설정
# -----------------------------------------------------------------------------
_BASE_DIR = Path(__file__).resolve().parent
# 프로젝트 루트 기준: data_processed/processed/merged/heatguard_ranking_data_cluster_1.csv
_PROJECT_ROOT = _BASE_DIR.parent.parent
_DEFAULT_CSV = _PROJECT_ROOT / "data_processed" / "processed" / "merged" / "heatguard_ranking_data_cluster_1.csv"

MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def z_to_interp(z: float, high_label: str = "높음", low_label: str = "낮음") -> str:
    """z-score를 서울 평균 대비 해석 문구로 변환."""
    if z >= 1.0:
        return f"서울 평균보다 {high_label}"
    elif z >= 0.5:
        return f"서울 평균보다 다소 {high_label}"
    elif z <= -1.0:
        return f"서울 평균보다 {low_label}"
    elif z <= -0.5:
        return f"서울 평균보다 다소 {low_label}"
    else:
        return "서울 평균과 비슷함"


def build_prompt(row: pd.Series) -> str:
    """한 행(row)에 대해 LLM에 넘길 프롬프트 문자열을 만든다."""
    suhii_interp = z_to_interp(row["suhii_score_z"], "높음", "낮음")
    temp_interp = z_to_interp(row["apparent_temp_score_z"], "높음", "낮음")
    flow_interp = z_to_interp(row["bus_flow_score_z"], "높음", "낮음")
    ndvi_interp = z_to_interp(row["ndvi_score_z"], "높음", "낮음")
    vuln_interp = z_to_interp(row["vulnerable_score_z"], "높음", "낮음")
    ndvi_z = row["ndvi_score_z"]

    cluster_name = row.get("cluster_name", "")
    # cluster_subtype = row.get("cluster_subtype", "")
    # if cluster_subtype == "" or (isinstance(cluster_subtype, float) and pd.isna(cluster_subtype)):
    #     cluster_subtype = "해당 없음"

    prompt = f"""
당신은 도시 폭염 대응 전문가입니다.
아래 데이터를 바탕으로 이 위치에 쿨링포그 설치가 추천되는 이유를 3개의 간결한 문장으로 설명해주세요.

모든 문장은 반드시 명사형 서술로 끝내세요.
(예: ~높음, ~많음, ~큼, ~있음, ~집중됨, ~우려 큼, ~의미 있음 등)
'~합니다', '~됩니다', '~중요합니다'와 같은 설명형·보고서체 문장은 사용하지 마세요.

"체감온도를 낮춤", "열 스트레스를 완화"와 같은 일반적 효과 문구를 그대로 반복하지 마세요.
효과는 대상(보행자/주민/취약계층) 또는 상황(폭염 시/야외 체류 시)을 포함해 표현하세요.
지표 간 논리적으로 모순되는 설명은 작성하지 마세요.
(예: 체감온도가 낮은데 더위가 큼 등)

【위치 정보】
군집명이나 지역 유형, 하위 유형 명칭을 문장에 직접 사용하지 마세요.
대신 지표의 상태를 설명하여 유형을 암묵적으로 드러내세요.

- 이 위치는 "{cluster_name}" 유형의 지역입니다.
- 군집 및 하위 유형 정보는 내부 분류 참고용입니다.
- 설명에는 해당 명칭을 직접 사용하지 말고, 지표의 상태만 근거로 사용하세요.

- 지역: {row['gu_name']} {row['dong_name']}
- 종합 점수: {row['GlobalScore']:.3f} (순위: {int(row['Rank'])}위)

【주요 지표 (서울 평균 대비)】

※ 아래 지표 중 이 위치의 특성을 가장 잘 설명하는 핵심 지표 1~2개만 선택하여 설명하세요.
나머지 지표는 언급하지 않아도 됩니다.

- 열섬강도(SUHII): {suhii_interp} (z-score: {row['suhii_score_z']:.2f})
- 체감온도: {temp_interp} (z-score: {row['apparent_temp_score_z']:.2f})
- 유동인구: {flow_interp} (z-score: {row['bus_flow_score_z']:.2f})
- 녹지 현황: {ndvi_interp} (z-score: {ndvi_z:.2f})
- 취약계층 밀도: {vuln_interp} (z-score: {row['vulnerable_score_z']:.2f})

【출력 형식】
JSON 배열로만 출력하세요. 각 항목은 하나의 문장입니다.

마지막 문장은 쿨링포그의 효과를
'누구에게 / 어떤 상황에서 / 왜 의미가 있는지' 중 하나로 구체화하세요.

예시:
[
  "체감온도와 유동인구가 동시에 높아 폭염 노출이 집중되는 지역",
  "여름철 야외 체류 인구가 많아 건강 피해 우려가 큼",
  "보행자 활동 시간대의 열 스트레스를 줄이는 데 의미가 있음"
]

반드시 JSON 배열 형식으로만 출력하고, 다른 설명이나 마크다운은 포함하지 마세요."""
    return prompt.strip()


def call_llm(prompt: str) -> str:
    """OpenAI Chat Completions API로 프롬프트를 보내고 응답 텍스트를 반환한다."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY 환경 변수를 설정해 주세요.")
    client = OpenAI(api_key=api_key)
    resp = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
    )
    return (resp.choices[0].message.content or "").strip()


def parse_summary_response(raw: str) -> str:
    """
    LLM 응답에서 JSON 배열만 추출해, 그대로 문자열로 저장할 수 있게 반환한다.
    (나중에 파싱 가능하도록 JSON 배열 문자열로 저장)
    """
    raw = raw.strip()
    # ```json ... ``` 제거
    if "```" in raw:
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```\s*$", "", raw)
    # 첫 '[' 와 마지막 ']' 사이 추출
    start = raw.find("[")
    end = raw.rfind("]")
    if start == -1 or end == -1 or end <= start:
        return raw
    arr_str = raw[start : end + 1]
    try:
        arr = json.loads(arr_str)
        if isinstance(arr, list):
            return json.dumps(arr, ensure_ascii=False)
    except json.JSONDecodeError:
        pass
    return arr_str


def main() -> None:
    parser = argparse.ArgumentParser(
        description="heatguard_ranking_data_cluster_1.csv에 LLM_summary 컬럼 추가"
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=_DEFAULT_CSV,
        help="입력 CSV 경로 (기본: data_processed/processed/merged/heatguard_ranking_data_cluster_1.csv)",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=None,
        metavar="PATH",
        help="출력 CSV 경로 (지정하지 않으면 입력 파일명_llm_summary.csv 로 저장, 원본 덮어쓰지 않음)",
    )
    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument("--all", action="store_true", help="모든 행에 대해 LLM_summary 생성")
    group.add_argument(
        "--limit",
        type=int,
        metavar="N",
        default=None,
        help="테스트용: 상위 N개 행에만 (기본: 3)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="실제 API 호출 없이 프롬프트만 출력 (1행만)",
    )
    args = parser.parse_args()

    if not args.all and args.limit is None:
        args.limit = 3

    if not args.dry_run and not os.getenv("OPENAI_API_KEY"):
        print("OPENAI_API_KEY 환경 변수가 없습니다.", flush=True)
        print("PowerShell: $env:OPENAI_API_KEY = \"your-key\"", flush=True)
        print("CMD: set OPENAI_API_KEY=your-key", flush=True)
        if sys.platform == "win32" and sys.stdin.isatty():
            input("엔터를 누르면 종료합니다...")
        sys.exit(1)

    csv_path = args.csv.resolve()
    if not csv_path.is_file():
        raise FileNotFoundError(f"CSV 파일을 찾을 수 없습니다: {csv_path}")
    if args.output is not None:
        out_path = Path(args.output).resolve()
    else:
        stem = csv_path.stem
        out_path = csv_path.parent / (stem + "_llm_summary.csv")
    _debug("출력 파일: %s" % out_path)

    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    if "LLM_summary" not in df.columns:
        df["LLM_summary"] = ""

    if args.all:
        indices = df.index.tolist()
    else:
        n = max(1, args.limit)
        indices = df.index[:n].tolist()

    total = len(indices)
    _debug("처리 대상: %d건" % total)
    logger.info("처리 대상 행 수: %d", total)

    if args.dry_run:
        row = df.loc[indices[0]]
        logger.info("=== 1행 분 프롬프트 (dry-run) ===")
        print(build_prompt(row))
        return

    for i, idx in enumerate(indices):
        row = df.loc[idx]
        prompt = build_prompt(row)
        try:
            response = call_llm(prompt)
            summary = parse_summary_response(response)
            df.at[idx, "LLM_summary"] = summary
            logger.info("[%d/%d] index=%s 완료", i + 1, total, idx)
        except Exception as e:
            logger.exception("[%d/%d] index=%s 실패: %s", i + 1, total, idx, e)
            df.at[idx, "LLM_summary"] = f"[오류: {e!r}]"

    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False, encoding="utf-8-sig")
    _debug("저장 완료: %s" % out_path)
    logger.info("저장 완료: %s", out_path)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        _debug("예외: %s" % e)
        print("오류 발생: %s" % e, flush=True)
        if sys.platform == "win32" and sys.stdin.isatty():
            input("엔터를 누르면 종료합니다...")
        raise
