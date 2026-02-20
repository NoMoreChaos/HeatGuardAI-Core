import ast
from pathlib import Path

import pandas as pd

LLM_DIR = Path(__file__).resolve().parent.parent
INPUT_PATH = LLM_DIR / "heatguard_ranking_data_cluster_1_llm_summary.csv"
OUTPUT_PATH = LLM_DIR / "heatguard_ranking_data_cluster_1_llm_summary_postprocess.csv"
PREFIX = "이 지역은 "


def parse_and_clean_first_item(val):
    """LLM_summary 셀을 파싱하고, 첫 문자열이 '이 지역은 '으로 시작하면 접두사 제거."""
    if pd.isna(val):
        return val
    if isinstance(val, str):
        try:
            lst = ast.literal_eval(val)
        except (ValueError, SyntaxError):
            return val
    else:
        lst = val
    if not isinstance(lst, list) or len(lst) == 0:
        return lst
    first = lst[0]
    if isinstance(first, str) and first.startswith(PREFIX):
        lst = [first[len(PREFIX) :]] + list(lst[1:])
    return lst


def main():
    df = pd.read_csv(INPUT_PATH)
    df["LLM_summary"] = df["LLM_summary"].apply(parse_and_clean_first_item)
    df.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")
    print(f"저장 완료: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
