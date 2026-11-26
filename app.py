from pathlib import Path
import pickle
import random
import re
import difflib
import warnings

import numpy as np
import pandas as pd
import streamlit as st
import Orange

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "industry_data.csv"
MODEL_PATHS = {
    1: BASE_DIR / "models" / "model_1y.pkcls",
    2: BASE_DIR / "models" / "model_2y.pkcls",
    3: BASE_DIR / "models" / "model_3y.pkcls",
}
BASE_YEAR = 2025

st.set_page_config(
    page_title="업종 폐업률 예측 서비스",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
.block-container {max-width: 1050px; padding-top: 2.2rem; padding-bottom: 3rem;}
.hero {padding: 1.6rem 1.7rem; border-radius: 20px; background: linear-gradient(120deg,#132b45,#245b71); color: white; margin-bottom: 1.2rem;}
.hero h1 {font-size: 2rem; margin: 0 0 .5rem 0; color: white;}
.hero p {margin: 0; color: #e3edf3; font-size: 1rem;}
.small-note {color:#667085; font-size:.88rem;}
.metric-card {border:1px solid rgba(128,128,128,.25); border-radius:16px; padding:1.05rem; min-height:135px;}
.metric-label {font-size:.9rem; opacity:.75; margin-bottom:.45rem;}
.metric-value {font-size:2rem; font-weight:750; line-height:1.2;}
.keyword-note {font-size:.86rem; opacity:.72; margin-bottom:.4rem;}
div[data-testid="stButton"] button {border-radius: 12px;}
</style>
<div class="hero">
  <h1>업종 폐업률 예측 서비스</h1>
  <p>2025년 서울시 업종 데이터를 바탕으로 1년·2년·3년 후 업종별 폐업률을 추정합니다.</p>
</div>
""", unsafe_allow_html=True)

@st.cache_data

def load_data():
    data = pd.read_csv(DATA_PATH, encoding="utf-8-sig")
    data["업종명"] = data["업종명"].astype(str).str.strip()
    data = data[data["연도"] == BASE_YEAR].copy()
    if data.empty:
        raise ValueError("CSV에서 2025년 자료를 찾을 수 없습니다.")
    data = data.drop_duplicates(subset=["업종명"], keep="first")
    return data

@st.cache_resource

def load_models():
    loaded = {}
    for year, path in MODEL_PATHS.items():
        if not path.exists():
            raise FileNotFoundError(f"모델 파일을 찾을 수 없습니다: {path.name}")
        # PKCLS는 사용자가 제공한 Orange 모델 파일입니다.
        with open(path, "rb") as f:
            loaded[year] = pickle.load(f)
    return loaded


def norm(text):
    return re.sub(r"[\s\-_·./()]+", "", str(text)).lower()


def get_alias_match(query, industries):
    q = norm(query)
    # 흔히 쓰는 표현을 공공데이터 업종명으로 연결하는 간단한 별칭 사전
    aliases = {
        "카페": "커피-음료", "카페창업": "커피-음료", "커피숍": "커피-음료", "커피전문점": "커피-음료",
        "피씨방": "PC방", "피시방": "PC방", "pc방": "PC방",
        "치킨집": "치킨전문점", "치킨": "치킨전문점",
        "빵집": "제과점", "베이커리": "제과점", "빵": "제과점",
        "편의점창업": "편의점", "슈퍼": "슈퍼마켓", "마트": "슈퍼마켓",
        "미용실": "미용실", "헤어샵": "미용실", "헤어숍": "미용실",
        "헬스장": "스포츠클럽", "헬스클럽": "스포츠클럽",
        "네일샵": "네일숍", "네일아트": "네일숍",
    }
    target = aliases.get(q)
    if target:
        for name in industries:
            if norm(name) == norm(target):
                return name
    for name in industries:
        if norm(name) == q:
            return name
    return None


def similar_industries(query, industries, limit=5):
    q = norm(query)
    scored = []
    for name in industries:
        n = norm(name)
        ratio = difflib.SequenceMatcher(None, q, n).ratio()
        if q and (q in n or n in q):
            ratio = max(ratio, 0.82)
        scored.append((ratio, name))
    return [name for score, name in sorted(scored, reverse=True)[:limit] if score >= 0.25]


def build_model_input(model, row, industry):
    """Build Orange Table matching the saved model's input domain."""
    domain = model.domain
    attributes = list(domain.attributes)
    values = []
    numeric_values = {
        "연도": BASE_YEAR,
        "전체_점포_수": float(row["전체_점포_수"]),
        "유사_업종_점포_수": float(row["유사_업종_점포_수"]),
        "개업률": float(row["개업률"]),
        "폐업률": float(row["폐업률"]),
    }
    for var in attributes:
        name = var.name
        if name in numeric_values:
            values.append(numeric_values[name])
        elif name.startswith("업종명="):
            category = name.split("=", 1)[1]
            values.append(1.0 if norm(category) == norm(industry) else 0.0)
        elif name == "업종명" and getattr(var, "is_discrete", False):
            categories = list(var.values)
            matched = next((i for i, v in enumerate(categories) if norm(v) == norm(industry)), None)
            values.append(float(matched) if matched is not None else np.nan)
        else:
            # 모델 도메인에 예기치 않은 변수가 있을 경우 기본값을 넣되 경고를 띄움
            values.append(0.0)

    X = np.asarray([values], dtype=float)
    class_vars = list(domain.class_vars)
    Y = np.full((1, len(class_vars)), np.nan, dtype=float) if class_vars else None
    prediction_domain = Orange.data.Domain(attributes, class_vars)
    return Orange.data.Table.from_numpy(prediction_domain, X, Y)


def predict_one(model, row, industry):
    table = build_model_input(model, row, industry)
    prediction = model(table)
    return float(np.asarray(prediction).reshape(-1)[0])

try:
    data_2025 = load_data()
    models = load_models()
except Exception as exc:
    st.error("데이터 또는 Orange 모델을 불러오지 못했습니다.")
    st.exception(exc)
    st.stop()

industries = sorted(data_2025["업종명"].dropna().unique().tolist())

if "suggestions" not in st.session_state:
    st.session_state.suggestions = random.sample(industries, min(5, len(industries)))
if "industry_query" not in st.session_state:
    st.session_state.industry_query = ""

st.subheader("🔎 어떤 업종을 창업할 계획인가요?")
st.markdown('<div class="keyword-note">검색 전 추천 키워드 · 버튼을 누르면 검색창에 자동 입력됩니다.</div>', unsafe_allow_html=True)
cols = st.columns(5)
for i, name in enumerate(st.session_state.suggestions):
    with cols[i]:
        if st.button(name, key=f"suggest_{i}", use_container_width=True):
            st.session_state.industry_query = name
            st.rerun()

c1, c2 = st.columns([5, 1])
with c1:
    query = st.text_input("업종명", key="industry_query", placeholder="예: 카페, PC방, 미용실, 치킨집")
with c2:
    st.write("")
    st.write("")
    if st.button("추천 새로고침", use_container_width=True):
        st.session_state.suggestions = random.sample(industries, min(5, len(industries)))
        st.rerun()

submitted = st.button("폐업률 예측하기", type="primary", use_container_width=True)

if submitted:
    st.session_state.pop("selected_industry", None)
    if not query.strip():
        st.warning("분석할 업종명을 입력하거나 추천 키워드를 선택해 주세요.")
    else:
        matched = get_alias_match(query, industries)
        if matched is None:
            candidates = similar_industries(query, industries)
            if not candidates:
                st.error("자료에서 비슷한 업종을 찾지 못했습니다. 다른 업종명으로 검색해 주세요.")
            else:
                st.info("입력한 업종과 정확히 일치하는 항목은 없어, 자료에 있는 유사 업종을 찾았습니다.")
                chosen = st.selectbox("분석할 유사 업종을 선택하세요.", candidates, key="candidate_choice")
                if st.button("선택한 업종 분석", type="primary"):
                    st.session_state.selected_industry = chosen
                    st.rerun()
        else:
            st.session_state.selected_industry = matched

if st.session_state.get("selected_industry"):
    selected = st.session_state.selected_industry
    match_row = data_2025[data_2025["업종명"] == selected]
    if match_row.empty:
        st.error("선택한 업종의 2025년 데이터를 찾을 수 없습니다.")
    else:
        row = match_row.iloc[0]
        try:
            results = {h: predict_one(models[h], row, selected) for h in (1, 2, 3)}
            results = {h: min(100.0, max(0.0, v)) for h, v in results.items()}
            st.divider()
            st.subheader(f"📈 {selected} 업종 예측 결과")
            st.caption("기준 데이터: 2025년 서울시 전체 업종 통합 자료")
            cards = st.columns(3)
            for col, horizon in zip(cards, (1, 2, 3)):
                with col:
                    st.markdown(f'''<div class="metric-card"><div class="metric-label">{horizon}년 후 예상 폐업률</div><div class="metric-value">{results[horizon]:.2f}%</div></div>''', unsafe_allow_html=True)
            chart_df = pd.DataFrame({"예측 기간": ["1년 후", "2년 후", "3년 후"], "예상 폐업률(%)": [results[1], results[2], results[3]]}).set_index("예측 기간")
            st.markdown("#### 기간별 예측 비교")
            st.bar_chart(chart_df, y="예상 폐업률(%)", use_container_width=True)
            st.markdown("#### 분석에 사용한 2025년 업종 데이터")
            summary = pd.DataFrame({
                "항목": ["전체 점포 수", "유사 업종 점포 수", "개업률", "폐업률"],
                "값": [f"{row['전체_점포_수']:,.2f}개", f"{row['유사_업종_점포_수']:,.2f}개", f"{row['개업률']:.2f}%", f"{row['폐업률']:.2f}%"],
            })
            st.table(summary)
            st.info("이 결과는 업종 단위의 과거 통계와 Random Forest 모델을 이용한 폐업률 추정치입니다. 개별 신규 점포가 폐업할 확률이나 폐업 여부를 보장하는 값은 아닙니다.")
            st.warning("1년 후 모델은 2025년 행의 실제 1년 후 폐업률도 학습에 포함했을 수 있어, 2025년 기준 예측은 학습 데이터에 대한 예측일 가능성이 있습니다. 정확한 미래 성능을 확인하려면 2025년을 학습에서 제외한 별도 검증이 필요합니다.")
        except Exception as exc:
            st.error("모델 예측 과정에서 오류가 발생했습니다. Orange 모델의 버전 또는 입력 특성 구성을 확인해야 합니다.")
            st.exception(exc)

st.divider()
st.markdown("**분석 기준 및 한계**")
st.markdown("- 이 서비스는 2025년의 업종별 점포 수·유사 업종 점포 수·개업률·폐업률을 모델 입력으로 사용합니다.\n- 업종명이 데이터에 없으면 유사한 업종명을 제안하며, 사용자가 최종 분석 대상을 선택합니다.\n- 2년 후·3년 후 예측은 과거에 해당 미래 관측값이 존재하는 연도들의 패턴을 학습한 결과입니다.\n- 예측값은 과거 데이터의 범위와 품질에 영향을 받으며, 실제 시장 상황을 보장하지 않습니다.")
