import json
import os
import datetime as dt   # ✅ eklendi
import streamlit as st

# -------------------------------
# GENERAL SETTINGS
# -------------------------------
st.set_page_config(page_title="Data Interview Christmas Tree", page_icon="🎄")

NUM_QUESTIONS = 30
DATA_FILE = "answers.json"

# Challenge başlangıç tarihi (kendine göre değiştir)
CHALLENGE_START = dt.date(2025, 12, 3)

# -------------------------------
# FUNCTIONS
# -------------------------------
def load_data():
    if os.path.exists(DATA_FILE):
        try:
            with open(DATA_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}

def save_data(data):
    with open(DATA_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def get_current_day(answers_list, max_open_day):
    """
    İlk cevapsız sorunun gün numarasını döndürür,
    ama sadece max_open_day'e kadar bakar.
    Hepsi doluysa None.
    """
    for i in range(max_open_day):
        if answers_list[i] is None:
            return i + 1
    return None  # bugüne kadar olan soruların hepsi cevaplanmış

# -------------------------------
# QUESTIONS and ANSWERS
# -------------------------------
QUESTIONS = {
    1: (
       "A researcher wants to estimate the average customer satisfaction score (on a scale of 1–10) for a product.\n After collecting a sample, they compute a 95% confidence interval of 7.2 ± 0.4.\n What does this confidence interval represent?\n\n"
        "- **A)** It means that the true average is definitely 7.2.\n"
        "- **B)** If we were to repeat the sampling process, all sample means would fall between 7.2 and 7.6.\n"
        "- **C)** If we were to repeat this sampling procedure many times, 95% of the confidence intervals constructed would contain the true population mean.\n"
        "- **D)** It means that 95% of the individuals in the sample scored between 7.2 and 7.6.\n"
    ),
    2: (
        "“______ is a common method used to handle categorical data in preprocessing.”\n\n"
        "- **A)** One-Hot Encoding\n"
        "- **B)** Principal Component Analysis\n"
        "- **C)** Logistic Regression\n"
        "- **D)** Feature Scaling\n"
    ),
    3: (
        "“______ is a common method used to handle categorical data in preprocessing.”\n\n"
        "- **A)** One-Hot Encoding\n"
        "- **B)** Principal Component Analysis\n"
        "- **C)** Logistic Regression\n"
        "- **D)** Feature Scaling\n"
    ),
    4: (
        "“______ is a common method used to handle categorical data in preprocessing.”\n\n"
        "- **A)** One-Hot Encoding\n"
        "- **B)** Principal Component Analysis\n"
        "- **C)** Logistic Regression\n"
        "- **D)** Feature Scaling\n"
    ),
    5: (
        "“______ is a common method used to handle categorical data in preprocessing.”\n\n"
        "- **A)** One-Hot Encoding\n"
        "- **B)** Principal Component Analysis\n"
        "- **C)** Logistic Regression\n"
        "- **D)** Feature Scaling\n"
    ),
    6: (
        "“______ is a common method used to handle categorical data in preprocessing.”\n\n"
        "- **A)** One-Hot Encoding\n"
        "- **B)** Principal Component Analysis\n"
        "- **C)** Logistic Regression\n"
        "- **D)** Feature Scaling\n"
    ),
    7: (
        "“______ is a common method used to handle categorical data in preprocessing.”\n\n"
        "- **A)** One-Hot Encoding\n"
        "- **B)** Principal Component Analysis\n"
        "- **C)** Logistic Regression\n"
        "- **D)** Feature Scaling\n"
    ),
    8: (
        "“______ is a common method used to handle categorical data in preprocessing.”\n\n"
        "- **A)** One-Hot Encoding\n"
        "- **B)** Principal Component Analysis\n"
        "- **C)** Logistic Regression\n"
        "- **D)** Feature Scaling\n"
    ),
    9: (
        "“______ is a common method used to handle categorical data in preprocessing.”\n\n"
        "- **A)** One-Hot Encoding\n"
        "- **B)** Principal Component Analysis\n"
        "- **C)** Logistic Regression\n"
        "- **D)** Feature Scaling\n"
    ),
    10: (
        "“______ is a common method used to handle categorical data in preprocessing.”\n\n"
        "- **A)** One-Hot Encoding\n"
        "- **B)** Principal Component Analysis\n"
        "- **C)** Logistic Regression\n"
        "- **D)** Feature Scaling\n"
    ),
    11: (
        "“______ is a common method used to handle categorical data in preprocessing.”\n\n"
        "- **A)** One-Hot Encoding\n"
        "- **B)** Principal Component Analysis\n"
        "- **C)** Logistic Regression\n"
        "- **D)** Feature Scaling\n"
    ),
    12: (
        "“______ is a common method used to handle categorical data in preprocessing.”\n\n"
        "- **A)** One-Hot Encoding\n"
        "- **B)** Principal Component Analysis\n"
        "- **C)** Logistic Regression\n"
        "- **D)** Feature Scaling\n"
    ),
    13: (
        "“______ is a common method used to handle categorical data in preprocessing.”\n\n"
        "- **A)** One-Hot Encoding\n"
        "- **B)** Principal Component Analysis\n"
        "- **C)** Logistic Regression\n"
        "- **D)** Feature Scaling\n"
    ),
    14: (
        "“______ is a common method used to handle categorical data in preprocessing.”\n\n"
        "- **A)** One-Hot Encoding\n"
        "- **B)** Principal Component Analysis\n"
        "- **C)** Logistic Regression\n"
        "- **D)** Feature Scaling\n"
    ),
    15: (
        "“______ is a common method used to handle categorical data in preprocessing.”\n\n"
        "- **A)** One-Hot Encoding\n"
        "- **B)** Principal Component Analysis\n"
        "- **C)** Logistic Regression\n"
        "- **D)** Feature Scaling\n"
    ),
    16: (
        "“______ is a common method used to handle categorical data in preprocessing.”\n\n"
        "- **A)** One-Hot Encoding\n"
        "- **B)** Principal Component Analysis\n"
        "- **C)** Logistic Regression\n"
        "- **D)** Feature Scaling\n"
    ),
    17: (
        "“______ is a common method used to handle categorical data in preprocessing.”\n\n"
        "- **A)** One-Hot Encoding\n"
        "- **B)** Principal Component Analysis\n"
        "- **C)** Logistic Regression\n"
        "- **D)** Feature Scaling\n"
    ),
    18: (
        "“______ is a common method used to handle categorical data in preprocessing.”\n\n"
        "- **A)** One-Hot Encoding\n"
        "- **B)** Principal Component Analysis\n"
        "- **C)** Logistic Regression\n"
        "- **D)** Feature Scaling\n"
    ),
    19: (
        "“______ is a common method used to handle categorical data in preprocessing.”\n\n"
        "- **A)** One-Hot Encoding\n"
        "- **B)** Principal Component Analysis\n"
        "- **C)** Logistic Regression\n"
        "- **D)** Feature Scaling\n"
    ),
    20: (
        "“______ is a common method used to handle categorical data in preprocessing.”\n\n"
        "- **A)** One-Hot Encoding\n"
        "- **B)** Principal Component Analysis\n"
        "- **C)** Logistic Regression\n"
        "- **D)** Feature Scaling\n"
    ),
    21: (
        "“______ is a common method used to handle categorical data in preprocessing.”\n\n"
        "- **A)** One-Hot Encoding\n"
        "- **B)** Principal Component Analysis\n"
        "- **C)** Logistic Regression\n"
        "- **D)** Feature Scaling\n"
    ),
    22: (
        "“______ is a common method used to handle categorical data in preprocessing.”\n\n"
        "- **A)** One-Hot Encoding\n"
        "- **B)** Principal Component Analysis\n"
        "- **C)** Logistic Regression\n"
        "- **D)** Feature Scaling\n"
    ),
    23: (
        "“______ is a common method used to handle categorical data in preprocessing.”\n\n"
        "- **A)** One-Hot Encoding\n"
        "- **B)** Principal Component Analysis\n"
        "- **C)** Logistic Regression\n"
        "- **D)** Feature Scaling\n"
    ),
    24: (
        "“______ is a common method used to handle categorical data in preprocessing.”\n\n"
        "- **A)** One-Hot Encoding\n"
        "- **B)** Principal Component Analysis\n"
        "- **C)** Logistic Regression\n"
        "- **D)** Feature Scaling\n"
    ),
    25: (
        "“______ is a common method used to handle categorical data in preprocessing.”\n\n"
        "- **A)** One-Hot Encoding\n"
        "- **B)** Principal Component Analysis\n"
        "- **C)** Logistic Regression\n"
        "- **D)** Feature Scaling\n"
    ),
    26: (
        "“______ is a common method used to handle categorical data in preprocessing.”\n\n"
        "- **A)** One-Hot Encoding\n"
        "- **B)** Principal Component Analysis\n"
        "- **C)** Logistic Regression\n"
        "- **D)** Feature Scaling\n"
    ),
    27: (
        "“______ is a common method used to handle categorical data in preprocessing.”\n\n"
        "- **A)** One-Hot Encoding\n"
        "- **B)** Principal Component Analysis\n"
        "- **C)** Logistic Regression\n"
        "- **D)** Feature Scaling\n"
    ),
    28: (
        "“______ is a common method used to handle categorical data in preprocessing.”\n\n"
        "- **A)** One-Hot Encoding\n"
        "- **B)** Principal Component Analysis\n"
        "- **C)** Logistic Regression\n"
        "- **D)** Feature Scaling\n"
    ),
    29: (
        "“______ is a common method used to handle categorical data in preprocessing.”\n\n"
        "- **A)** One-Hot Encoding\n"
        "- **B)** Principal Component Analysis\n"
        "- **C)** Logistic Regression\n"
        "- **D)** Feature Scaling\n"
    ),
    30: (
        "“______ is a common method used to handle categorical data in preprocessing.”\n\n"
        "- **A)** One-Hot Encoding\n"
        "- **B)** Principal Component Analysis\n"
        "- **C)** Logistic Regression\n"
        "- **D)** Feature Scaling\n"
    )
}

CORRECT_OPTIONS = {i: "A" for i in range(1, 31)}

# -------------------------------
# USER
# -------------------------------
st.sidebar.header("User")
username = st.sidebar.text_input("Username")

if not username:
    st.warning("Enter a username on the left to continue.")
    st.stop()

data = load_data()
user_record = data.get(username, {"answers": [None] * NUM_QUESTIONS})
answers = user_record["answers"]

# --- BUGÜN EN FAZLA HANGİ GÜNE KADAR SORU AÇILABİLİR? ---
today = dt.date.today()
days_since_start = (today - CHALLENGE_START).days

if days_since_start < 0:
    max_open_day = 1
else:
    max_open_day = min(NUM_QUESTIONS, days_since_start + 1)

current_day = get_current_day(answers, max_open_day)

if current_day is None:
    st.sidebar.markdown(
        f"✅ All questions up to day {max_open_day} are completed! 🎉"
    )
else:
    st.sidebar.markdown(f"Today's question: **{current_day}**")

# -------------------------------
# TREE STRUCTURE
# -------------------------------
trunk_row = [1, 2]   # stem

# Leaves indexes (bottom to top, sol→sağ)
idx = 3
foliage_counts_bottom_up = [7, 6, 5, 4, 3, 2, 1]   # total 28
foliage_rows_bottom_up = []
for count in foliage_counts_bottom_up:
    row_indices = list(range(idx, idx + count))
    foliage_rows_bottom_up.append(row_indices)
    idx += count

# -------------------------------
# AĞAÇ OLUŞTURMA (HTML + CSS GRID)
# -------------------------------
def cell_class(cell_index, answers_list):
    """For HTML: CSS class based on cell type and response."""
    state = None
    if 1 <= cell_index <= NUM_QUESTIONS:
        state = answers_list[cell_index - 1]  # True / False / None

    if cell_index in trunk_row:
        base = "stem"
    else:
        base = "leaf"

    if state is True:
        status = "correct"
    elif state is False:
        status = "wrong"
    else:
        status = "empty"

    return f"{base} {status}"


def generate_tree_html(answers_list):
    # CSS styles
    style = """
    <style>
    .tree-wrapper {
        display: flex;
        flex-direction: column;
        align-items: center;
        padding: 16px 0;
    }
    .tree-row {
        display: flex;
        justify-content: center;
        gap: 6px;
        margin: 2px 0;
    }
    .tree-cell {
        width: 18px;
        height: 18px;
        border-radius: 4px;
        background: #f1ecff;
        box-shadow: 0 1px 2px rgba(0,0,0,0.12);
        position: relative;
        overflow: visible;
    }

    /* YILDIZ (en tepe) */
    .tree-row.star-row {
        margin-bottom: 4px;
    }
    .tree-star-wrapper {
        width: 18px;
        height: 18px;
        display: flex;
        align-items: center;
        justify-content: center;
    }
    .tree-star {
        font-size: 18px;
        line-height: 1;
        animation: star-glow 1.4s ease-in-out infinite alternate;
        filter: drop-shadow(0 0 4px #fffde7);
    }

    @keyframes star-glow {
        0% {
            transform: translateY(0px) scale(1);
            text-shadow:
                0 0 3px #fff9c4,
                0 0 6px #fff176,
                0 0 10px #ffd54f;
        }
        100% {
            transform: translateY(-1px) scale(1.05);
            text-shadow:
                0 0 6px #fff9c4,
                0 0 10px #fff176,
                0 0 16px #ffeb3b;
        }
    }

    /* Yaprak temel renkleri */
    .tree-cell.leaf.correct { background: #4caf50; }
    .tree-cell.leaf.wrong   { background: #e53935; }
    .tree-cell.leaf.empty   { background: #e9ddff; }

    /* Gövde renkleri */
    .tree-cell.stem.correct { background: #795548; }
    .tree-cell.stem.wrong   { background: #212121; }
    .tree-cell.stem.empty   { background: #bdbdbd; }

    /* 🎄 KUTU İÇİ MOR YILBAŞI TOPLARI (tüm yaprak kutuları için) */
    .tree-cell.leaf::before {
        content: "🔮";
        position: absolute;
        top: -1px;
        left: 50%;
        transform: translateX(-50%);
        font-size: 12px;
        text-shadow: 0 0 2px rgba(0,0,0,0.25);
    }

    </style>
    """

    rows_html = []

    # 0) YILDIZ SATIRI (ağacın en tepesi)
    star_row = """
    <div class="tree-row star-row">
        <div class="tree-star-wrapper">
            <div class="tree-star">⭐</div>
        </div>
    </div>
    """
    rows_html.append(star_row)

    # 1) Leaves:
    for row_indices in reversed(foliage_rows_bottom_up):
        cells = []
        for idx in row_indices:
            classes = cell_class(idx, answers_list)
            cells.append(f'<div class="tree-cell {classes}"></div>')
        rows_html.append('<div class="tree-row">' + "".join(cells) + '</div>')

    # 2) Stem:
    stem_cells = []
    for idx in trunk_row:
        classes = cell_class(idx, answers_list)
        stem_cells.append(f'<div class="tree-cell {classes}"></div>')
    rows_html.append('<div class="tree-row">' + "".join(stem_cells) + '</div>')

    html = style + '<div class="tree-wrapper">' + "".join(rows_html) + '</div>'
    return html


# -------------------------------
# TITLE AND TREE IMAGE
# -------------------------------
st.title("🎄 30 Days of Data Interview – Christmas Tree Tracker")

st.markdown(
    f"**User:** `{username}`  \n"
    f"Total questions: {NUM_QUESTIONS}  \n"
)

st.subheader("Tree Status")

# 👉 AĞAÇ İÇİN PLACEHOLDER
tree_placeholder = st.empty()
tree_placeholder.markdown(generate_tree_html(answers), unsafe_allow_html=True)

st.markdown(
    """
    **Colors:**

    - Leaves (questions 3–30): ✅ True = 🟩, ❌ False = 🟥  
    - Stem (questions 1–2): ✅ True = 🟫, ❌ False = ⬛
    """
)

# -------------------------------
# TODAY'S QUESTION
# -------------------------------
if current_day is None:
    st.subheader("All questions for today are completed 🎉")
    user_answer = None
else:
    st.subheader(f"Day {current_day} – Today's Question")

    question_text = QUESTIONS.get(
        current_day,
        "The question for today has not yet been defined."
    )
    st.write(question_text)

    user_answer = st.text_input("Your answer (e.g. A, B, C, D)", max_chars=10)

# -------------------------------
# SUBMIT BUTTON
# -------------------------------
if current_day is not None and st.button("Submit"):
    normalized = None
    if user_answer and user_answer.strip():
        normalized = user_answer.strip()[0].upper()

    correct_option = CORRECT_OPTIONS.get(current_day)

    if correct_option is None:
        st.info("The correct answer for this question has not yet been defined.")
    else:
        is_correct = (normalized == correct_option.upper())
        answers[current_day - 1] = is_correct

        user_record["answers"] = answers
        data[username] = user_record
        save_data(data)

        st.success("Your answer has been saved! 🎉 Tree above has been updated.")

        # 🔁 AĞACI AYNI ÇALIŞTIRMADA TEKRAR ÇİZDİR
        tree_placeholder.markdown(generate_tree_html(answers), unsafe_allow_html=True)
