import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path

st.set_page_config(page_title="송설 수학 플랫폼", page_icon="92", layout="centered")

# ---------------------------
# 기본 가중치
# ---------------------------
BASE_DIFF_WEIGHTS = {"하": 1.0, "중": 0.7, "상": 0.4}

TYPE_WEIGHT_INC_ON_WRONG = 1.20
TYPE_WEIGHT_DEC_ON_RIGHT = 0.95
TYPE_WEIGHT_MIN, TYPE_WEIGHT_MAX = 0.4, 3.0

ACC_ATTEMPT_MIN = 5
ACC_THRESHOLD   = 0.80
HARD_DIFF_BONUS = 0.40

RECENCY_PENALTY = 0.85

IMG_EXTS = (".png", ".jpg", ".jpeg", ".gif", ".webp")


# ---------------------------
# 문자열 정규화
# ---------------------------
def norm(s):
    return str(s).strip().lower()


# ---------------------------
# CSV 로드
# ---------------------------
@st.cache_data
def load_questions(csv_path):
    df = pd.read_csv(csv_path)
    required = {"id","type","difficulty","image","answer","explanation"}
    if not required.issubset(df.columns):
        raise ValueError("CSV 오류: 필요한 컬럼 없음")
    df = df.astype(str)
    return df


# ---------------------------
# 세션 상태 초기화
# ---------------------------
def init_state(df):
    if "type_weights" not in st.session_state:
        st.session_state.type_weights = {t:1.0 for t in df["type"].unique()}
    if "stats" not in st.session_state:
        st.session_state.stats = {
            t:{"attempts":0,"correct":0} for t in df["type"].unique()
        }
    if "history" not in st.session_state:
        st.session_state.history = []
    if "current_qid" not in st.session_state:
        st.session_state.current_qid = None
    if "feedback" not in st.session_state:
        st.session_state.feedback = None
    if "weight_used" not in st.session_state:
        st.session_state.weight_used = False
    if "answer_input" not in st.session_state:
        st.session_state.answer_input = ""


# ---------------------------
# 난이도 가중치 계산
# ---------------------------
def difficulty_weights_for_type(typ):
    base = BASE_DIFF_WEIGHTS.copy()
    stat = st.session_state.stats[typ]

    attempts, correct = stat["attempts"], stat["correct"]
    acc = correct/attempts if attempts > 0 else 0

    if attempts >= ACC_ATTEMPT_MIN and acc >= ACC_THRESHOLD:
        base["상"] += HARD_DIFF_BONUS

    return base


# ---------------------------
# 다음 문제 선택
# ---------------------------
def choose_next_question(df):
    scores = []
    last_qid = st.session_state.history[-1][0] if st.session_state.history else None

    for _, row in df.iterrows():
        type_w = st.session_state.type_weights[row["type"]]
        diff_w = difficulty_weights_for_type(row["type"])[row["difficulty"]]

        score = type_w * diff_w

        if last_qid and row["id"] == last_qid:
            score *= RECENCY_PENALTY

        if not Path(row["image"]).exists():
            score = 0

        scores.append(score)

    scores = np.array(scores, float)

    if scores.sum() == 0:
        return None

    probs = scores / scores.sum()
    idx = np.random.choice(len(df), p=probs)

    return df.iloc[idx]["id"]


# ---------------------------
# 가중치 반영
# ---------------------------
def apply_weight(q, correct):
    typ = q["type"]

    st.session_state.stats[typ]["attempts"] += 1
    if correct:
        st.session_state.stats[typ]["correct"] += 1

    w = st.session_state.type_weights[typ]
    w *= TYPE_WEIGHT_DEC_ON_RIGHT if correct else TYPE_WEIGHT_INC_ON_WRONG
    st.session_state.type_weights[typ] = max(TYPE_WEIGHT_MIN, min(TYPE_WEIGHT_MAX, w))


# ============================================================
# UI 시작
# ============================================================

st.title("송설 수학 플랫폼")

csv_path = st.text_input("CSV 경로", "questions.csv")

try:
    df = load_questions(csv_path)
except Exception as e:
    st.error(f"CSV 오류: {e}")
    st.stop()

init_state(df)

# ✅ 첫 문제 초기화 시 버튼 무반응 문제 해결 (강제 rerun)
if st.session_state.current_qid is None:
    st.session_state.current_qid = choose_next_question(df)
    st.session_state.weight_used = False
    st.session_state.feedback = None
    st.rerun()


# ---------------------------
# 가중치/정확도 표시
# ---------------------------
with st.expander("📊 현재 가중치 / 정확도"):
    st.table(pd.DataFrame({"weight": st.session_state.type_weights}).T)

    rows=[]
    for t,s in st.session_state.stats.items():
        att,cor = s["attempts"], s["correct"]
        acc = f"{(cor/att*100):.0f}%" if att else "-"
        rows.append([t,att,cor,acc])

    st.table(pd.DataFrame(rows, columns=["type","attempts","correct","accuracy"]))


# ---------------------------
# 현재 문제 표시
# ---------------------------
q = df[df["id"] == st.session_state.current_qid].iloc[0]

st.subheader("문제")

if Path(q["image"]).exists():
    st.image(q["image"], use_container_width=True)
else:
    st.warning("문제 이미지 없음!")

if st.session_state.feedback:
    st.info(st.session_state.feedback)


# ---------------------------
# 정답 입력
# ---------------------------
user_input = st.text_input("정답을 입력하세요", key="answer_input")
user_norm = norm(user_input)

# ✅ 여러 정답 허용 (예: 5|270)
answer_list = [norm(a) for a in q["answer"].split("|")]
input_is_correct = (user_norm in answer_list and user_norm != "")


# ---------------------------
# 채점 함수
# ---------------------------
def handle_check():
    correct = (user_norm in answer_list)

    if not st.session_state.weight_used:
        apply_weight(q, correct)
        st.session_state.history.append((q["id"], correct))
        st.session_state.weight_used = True

    st.session_state.feedback = "✅ 정답입니다!" if correct else "❌ 오답입니다!"
    st.rerun()


# ---------------------------
# 버튼 UI
# ---------------------------
col1, col2 = st.columns(2)

# ✅ 채점하기: 항상 표시
if col1.button("채점하기"):
    handle_check()

# ✅ 정답 입력하면 → 건너뛰기 버튼이 “다음문제”로 자동 전환
if input_is_correct:
    if col2.button("다음 문제"):
        st.session_state.pop("answer_input", None)
        st.session_state.current_qid = choose_next_question(df)
        st.session_state.weight_used = False
        st.session_state.feedback = None
        st.rerun()
else:
    if col2.button("건너뛰기"):
        if not st.session_state.weight_used:
            apply_weight(q, False)
            st.session_state.history.append((q["id"], False))
        st.session_state.feedback = "⏭️ 건너뛰었습니다."
        st.session_state.pop("answer_input", None)
        st.session_state.current_qid = choose_next_question(df)
        st.session_state.weight_used = False
        st.rerun()


# ---------------------------
# 해설
# ---------------------------
with st.expander("📝 해설 보기"):
    exp = q["explanation"]
    p = Path(exp)
    if p.exists() and p.suffix.lower() in IMG_EXTS:
        st.image(str(p), use_container_width=True)
    else:
        st.write(exp)
