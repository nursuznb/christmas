import json
import os
import streamlit as st

# -------------------------------
# GENERAL SETTINGS
# -------------------------------
st.set_page_config(page_title="Data Interview Christmas Tree", page_icon="🎄")

NUM_QUESTIONS = 30
DATA_FILE = "answers.json"

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

def get_current_day(answers_list):
    """İlk cevapsız sorunun gün numarasını döndürür. Hepsi doluysa None."""
    for i, a in enumerate(answers_list):
        if a is None:
            return i + 1
    return None  # tüm sorular cevaplanmış

# -------------------------------
# QUESTIONS and ANSWERS
# -------------------------------
QUESTIONS ={
   1: (
        "“______ is a common method used to handle categorical data in preprocessing.”\n\n"
        "- **A)** One-Hot Encoding\n"
        "- **B)** Principal Component Analysis\n"
        "- **C)** Logistic Regression\n"
        "- **D)** Feature Scaling\n"
  ),
    #2: (
      # "Which of the following is NOT a supervised learning algorithm?\n\n"
      # "A) Linear Regression\n\n"
     #   "B) K-Means Clustering\n\n"
      #  "C) Random Forest\n\n"
      #  "D) Logistic Regression\n\n"
    #),
    #3: (
       # "What does regularization do in a model?\n"
       # "A) Increases training error\n"
       # "B) Increases model complexity\n"
       # "C) Penalizes large weights to reduce overfitting\n"
        #"D) Removes features randomly\n"
    #),
    # 4: "Gün 4 sorusu...",
    # ...
    # 30: "Gün 30 sorusu..."
}

CORRECT_OPTIONS = {
   1: "A",   
   # 2: "B",   
    #3: "C",
    # 4: "A",
    # ...
    # 30: "D",
}


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

current_day = get_current_day(answers)

if current_day is None:
    st.sidebar.markdown("✅ All questions completed! 🎉")
else:
    st.sidebar.markdown(f"Today's question: **{current_day}**")

# -------------------------------
# TREE STRUCTURE
# -------------------------------
# 1–2: stem (1 row × 2 column)
# 3–30: leaves (28 box, triangle)

trunk_row = [1, 2]   # stem

# Leaves indexes (bottom to top, sol→sağleft to right)
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
def cell_emoji(cell_index, answers_list):
    """Color logic (we're using it just for clarification for now)."""
    state = None
    if 1 <= cell_index <= NUM_QUESTIONS:
        state = answers_list[cell_index - 1]

    # Stem (1–2)
    if cell_index in trunk_row:
        if state is True:
            return "🟫"
        elif state is False:
            return "⬛"
        else:
            return "⬜"

    # Leaves(3–30)
    if 3 <= cell_index <= NUM_QUESTIONS:
        if state is True:
            return "🟩"
        elif state is False:
            return "🟥"
        else:
            return "⬜"

    return "⬜"


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
    }

    /* Yaprak renkleri */
    .tree-cell.leaf.correct { background: #4caf50; }   /* yeşil */
    .tree-cell.leaf.wrong   { background: #e53935; }   /* kırmızı */
    .tree-cell.leaf.empty   { background: #e9ddff; }   /* lila / boş */

    /* Gövde renkleri */
    .tree-cell.stem.correct { background: #795548; }   /* kahverengi */
    .tree-cell.stem.wrong   { background: #212121; }   /* siyah */
    .tree-cell.stem.empty   { background: #bdbdbd; }   /* gri */
    </style>
    """

    rows_html = []

    # 1) Leaves:
    for row_indices in reversed(foliage_rows_bottom_up):  # alttan üste tanımlı → ters çevir
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
st.markdown(generate_tree_html(answers), unsafe_allow_html=True)

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
    st.subheader("All questions completed 🎉")
else:
    st.subheader(f"Day {current_day} – Today's Question")

    question_text = QUESTIONS.get(
        current_day,
        "The question for today has not yet been defined."
    )
    st.write(question_text)

    user_answer = st.text_input("Your answer (e.g. A, B, C, D)", max_chars=10)

    if st.button("Submit"):
        normalized = None
        if user_answer.strip():
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

            st.success("Your answer has been saved. The colors in the tree have been updated in the below. 🎄")

        st.subheader("Updated Tree")
        st.markdown(generate_tree_html(answers), unsafe_allow_html=True)
