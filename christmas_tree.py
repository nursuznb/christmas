import json
import os
import datetime as dt   
import streamlit as st

# -------------------------------
# GENERAL SETTINGS
# -------------------------------
st.set_page_config(page_title="Data Interview Christmas Tree", page_icon="🎄")

# 🔹 Google Analytics 
GA_SCRIPT = """
<!-- Google tag (gtag.js) -->
<script async src="https://www.googletagmanager.com/gtag/js?id=G-XXXXXXX"></script>
<script>
  window.dataLayer = window.dataLayer || [];
  function gtag(){dataLayer.push(arguments);}
  gtag('js', new Date());
  gtag('config', 'G-XXXXXXX');
</script>
"""

# HTML/JS
st.markdown(GA_SCRIPT, unsafe_allow_html=True)

NUM_QUESTIONS = 30
DATA_FILE = "answers.json"

# Challenge başlangıç tarihi 
CHALLENGE_START = dt.date(2025, 12, 2)

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
       "Say you are running a simple logistic regression to solve a problem but find the results to be unsatisfactory. What are some ways you might improve your model?\n\n"
       "I. Adding additional features\n\n"
       "II. Normalizing features\n\n"
       "III. Addressing outliers\n\n"
       "IV. Selecting variables\n\n"
       "V. Cross validation and hyperparameter tuning\n\n"
       
        "- **A)** III and IV\n"
        "- **B)** I, II, and III\n"
        "- **C)** II, III, and IV\n"
        "- **D)** All of the above\n"
    ),
    2: (
        "“______ is a common method used to handle categorical data in preprocessing.”\n\n"
        "- **A)** One-Hot Encoding\n"
        "- **B)** Principal Component Analysis\n"
        "- **C)** Logistic Regression\n"
        "- **D)** Feature Scaling\n"
    ),
    3: (
    "A researcher wants to estimate the average customer satisfaction score (on a scale of 1–10) for a product.\n"
    "After collecting a sample, they compute a 95% confidence interval of 7.2 ± 0.4.\n"
    "What does this confidence interval represent?\n\n"
    "- **A)** It means that the true average is definitely 7.2.\n"
    "- **B)** If we were to repeat the sampling process, all sample means would fall between 7.2 and 7.6.\n"
    "- **C)** If we were to repeat this sampling procedure many times, 95% of the confidence intervals constructed would contain the true population mean.\n"
    "- **D)** It means that 95% of the individuals in the sample scored between 7.2 and 7.6.\n"
    ),
    4: (
        "Say you flip a coin 10 times and observe only one head. What would be your p-value for testing whether the coin is fair or not?\n\n"
        "- **A)** 1/1024\n"
        "- **B)** 10/1024\n"
        "- **C)** 12/1024\n"
        "- **D)** 22/1024\n"
    ),
    5: (
        "What is an example of a machine learning algorithm that is not convex?\n\n"
        "- **A)** Ridge Regression\n"
        "- **B)** Neural Networks\n"
        "- **C)** Elastic Net Regression\n"
        "- **D)** Linear Support Vector Machine\n"
    ),
    6: (
        "Which of the following is NOT a common characteristic of both L1 and L2 regularization methods?\n\n"
        "- **A)** They help reduce overfitting.\n"
        "- **B)** They add a penalty term to the loss function.\n"
        "- **C)** They force all coefficients to become exactly zero.\n"
        "- **D)** They help control model complexity.\n"
    ),
    7: (
      "Say X is a univariate Gaussian random variable. What is the entropy of X?\n\n"
      "- **A)**  $H(X) = \\frac{1}{2} + \\log(\\sigma \\sqrt{2\\pi})$\n"
      "- **B)**  $H(X) = -\\log(\\sigma \\sqrt{2\\pi})$\n"
      "- **C)**  $H(X) = \\frac{1}{2} - \\log(\\sigma \\sqrt{2\\pi})$\n"
      "- **D)**  $H(X) = \\frac{1}{2}$\n"
    ),

   8: (
    "Say you were running a linear regression for a dataset but you accidentally duplicated every data point.\n"
    "What happens to your beta coefficient?\n"
    "- **A)**  $\\beta = (2X^T X)(2X^T y)$\n"
    "- **B)**  $\\beta = (2X^T X)^{-1}(2X^T y)$\n"
    "- **C)**  $\\beta = (2X^T X)^{-1}$\n"
    "- **D)**  $\\beta = (2X^T X)^{-1}(2X^T y)$\n"
),
    9: (
        "Say you are given a very large corpus of words. How would you identify synonyms?\n\n"
        "- **A)** Train word embeddings (e.g., Word2Vec) to represent words as vectors, then measure vector similarity (e.g., cosine similarity) or apply clustering / nearest-neighbor search to find similar words.\n"
        "- **B)** Count how often two words appear in the same sentence and consider any pair with similar frequencies as synonyms.\n"
        "- **C)** Sort all words alphabetically and compare words that start with the same letter to identify synonyms.\n"
        "- **D)** Remove all stopwords from the corpus and assume the remaining words that appear most frequently are each other’s synonyms.\n"
    ),
    10: (
        "Why is the Central Limit Theorem useful?\n\n"
        "- **A)** It guarantees that all random variables in nature follow a normal distribution regardless of sample size.\n"
        "- **B)** It states that any dataset becomes normally distributed after removing outliers.\n"
        "- **C)** It allows us to assume that the sampling distribution of the sample mean is approximately normal, even if the population distribution is not normal, provided the sample size is large enough.\n"
        "- **D)** It ensures that the variance of the sample mean is always equal to the variance of the population.\n"
    ),
    11: (
        "Which of the following best describes the fundamental difference between entropy and information gain?\n\n"
        "- **A)** Entropy can only be used with numerical variables, whereas information gain works only with categorical variables.\n"
        "- **B)** Entropy measures the uncertainty or impurity of a dataset, while information gain measures how much this uncertainty decreases after splitting based on a feature.\n"
        "- **C)** Information gain measures the raw disorder of the dataset, while entropy calculates the gain obtained after each split.\n"
        "- **D)** Entropy and information gain are the same concept and can be used interchangeably in decision tree algorithms.\n"
    ),
    12: (
        "In multivariate data analysis, _______ is a common technique for projecting high-dimensional data into two or three dimensions.\n\n"
        "- **A)** K-means clustering\n"
        "- **B)** Principal component analysis\n"
        "- **C)** t-Distributed ctochastic neighbor embedding (t-SNE)\n"
        "- **D)** Hierarchical clustering\n"
    ),
    13: (
        "How does autocorrelation help in time series analysis?\n\n"
        "- **A)** Autocorrelation measures the linear relationship between a time series and a lagged version of itself.\n"
        "- **B)** Autocorrelation helps in predicting future values of a time series based on its past values.\n"
        "- **C)** Autocorrelation measures the relationship between a time series and an unrelated external variable.\n"
        "- **D)** Autocorrelation is not relevant in time series analysis.\n"
    ),
    14: (
        "When applying LDA to a dataset, what assumption about the data does the data scientist need to consider for effective dimensionality reduction?\n\n"
        "- **A)** No assumption is needed\n"
        "- **B)** Homogeneity of covariance matricesn"
        "- **C)** Features are independent\n"
        "- **D)** Normal distribution of features\n"
    ),
    15: (
        "What is the significance of using a logarithmic scale in data visualization?\n\n"
        "- **A)** Displaying data in a bar chart format\n"
        "- **B)** Simplifying data representation for beginners\n"
        "- **C)** Emphasizing small values and compressing large values\n"
        "- **D)** Highlighting outliers in a dataset\n"
    ),
    16: (
        "What is the role of stationarity in time series forecasting?\n\n"
        "- **A)** Stationarity is not relevant in time series forecasting\n"
        "- **B)** Non-stationary time series are easier to forecast\n"
        "- **C)** Stationarity is only important for short-term forecasts\n"
        "- **D)** Stationarity ensures constant mean and variance over time\n"
    ),
    17: (
        "When dealing with high-dimensional data, _______ techniques are crucial to reduce complexity without losing significant information.\n\n"
        "- **A)** Dimensionality Reduction\n"
        "- **B)** Feature Engineering\n"
        "- **C)** Clustering\n"
        "- **D)** Ensemble Learning\n"
    ),
    18: (
        "What are the pros and cons of imputing missing data versus removing observations with missing data?\n\n"
        "- **A)** Faster processing, but decreased model performance\n"
        "- **B)** Reducing bias, but loss of information\n"
        "- **C)** Retaining more data for analysis, but potential introduction of bias\n"
        "- **D)** Increased accuracy, but longer processing time\n"
    ),
    19: (
        "How does the concept of 'manifold learning' relate to dimensionality reduction?\n\n"
        "- **A)** Emphasizes increasing the dimensionality of data for better understanding\n"
        "- **B)** Focuses on preserving the intrinsic geometry of high-dimensional data\n"
        "- **C)** Prioritizes random sampling for dimensionality reduction\n"
        "- **D)** Utilizes clustering algorithms to create high-dimensional representations\n"
    ),
    20: (
        "How does a confusion matrix help in evaluating the performance of a classification algorithm?\n\n"
        "- **A)** It visualizes the distribution of data points in a dataset\n"
        "- **B)** It evaluates the impact of outliers on model performance\n"
        "- **C)** It measures the accuracy of a model by comparing predicted and actual values\n"
        "- **D)** It provides a summary of prediction results, showing the correct and incorrect predictions\n"
    ),
    21: (
        "For a time-sensitive financial forecasting model, what considerations should be taken when choosing a supervised learning algorithm?\n\n"
        "- **A)** The algorithm's ability to handle streaming data\n"
        "- **B)** The algorithm's performance in terms of accuracy and precision\n"
        "- **C)** The algorithm's scalability to large datasets\n"
        "- **D)** The algorithm's interpretability for regulatory compliance\n"
    ),
    22: (
        "How does feature selection impact the performance of a machine learning model?\n\n"
        "- **A)** Increases model complexity, enhances generalization\n"
        "- **B)** Reduces overfitting, improves model training time\n"
        "- **C)** Has no impact on model performance\n"
        "- **D)** Improves model accuracy by default\n"
    ),
    23: (
        "In model evaluation, what does the AUC represent in the context of an ROC curve?\n\n"
        "- **A)** Accuracy of the model\n"
        "- **B)** True Positive Rate\n"
        "- **C)** Area Under the Curve\n"
        "- **D)** Probability of misclassification\n"
    ),
    24: (
        "In decision tree algorithms, the criterion of _______ is used to decide the feature on which to split.\n\n"
        "- **A)** Entropy\n"
        "- **B)** Gini index\n"
        "- **C)** Chi-square\n"
        "- **D)** Information gain\n"
    ),
    25: (
        "What is the significance of interaction terms in feature engineering?\n\n"
        "- **A)** Capturing synergistic effects between features\n"
        "- **B)** Eliminating outliers in the dataset\n"
        "- **C)** Improving model interpretability\n"
        "- **D)** Reducing dimensionality\n"
    ),
    26: (
        "In a medical diagnosis application, a supervised model shows high accuracy but low recall. What could be the implications for patient diagnosis?\n\n"
        "- **A)** The model is making many false positive predictions, leading to unnecessary treatments\n"
        "- **B)** The model is not suitable for medical diagnosis applications\n"
        "- **C)** The model is correctly identifying most positive cases but missing some, which could lead to false negatives\n"
        "- **D)** The model is well-balanced and is performing optimally\n"
    ),
    27: (
        "What is the bias-variance tradeoff in supervised learning, and why is it important?\n\n"
        "- **A)** One-Hot Encoding\n"
        "- **B)** Maximizing both bias and variance for a more flexible model\n"
        "- **C)** Ignoring bias and focusing solely on variance\n"
        "- **D)** Minimizing both bias and variance for optimal model performance\n"
    ),
    28: (
        "What is the main advantage of using a convolutional neural network (CNN) over a fully connected neural network for image processing?\n\n"
        "- **A)** Lower memory requirements\n"
        "- **B)** Better handling of sequential data\n"
        "- **C)** Faster training time\n"
        "- **D)** Ability to capture spatial hierarchies and local patterns\n"
    ),
    29: (
        "How do ensemble methods improve the performance of a supervised learning model?\n\n"
        "- **A)** Increasing model complexity\n"
        "- **B)** Applying feature scaling\n"
        "- **C)** Reducing the number of features\n"
        "- **D)** Averaging predictions of multiple models\n"
    ),
    30: (
        "What is the purpose of transfer learning in deep learning models?\n\n"
        "- **A)** Using knowledge gained from training on one task to improve performance on a different, but related, task\n"
        "- **B)** Ignoring pre-trained weights for better generalization\n"
        "- **C)** Training a model on multiple tasks simultaneously\n"
        "- **D)** Transferring weights directly from one layer to another in the same network\n"
    )
}

# SORULARIN DOĞRU ŞIKLARI
CORRECT_OPTIONS = {
    1: "D", 
    2: "A",
    3: "C",
    4: "D",
    5: "B",
    6: "C",
    7: "A",
    8: "B",
    9: "A",
    10: "C",
    11: "B",
    12: "C",
    13: "A",
    14: "B",
    15: "C",
    16: "D",
    17: "A",
    18: "C",
    19: "B",
    20: "D",
    21: "A",
    22: "B",
    23: "C",
    24: "D",
    25: "A",
    26: "C",
    27: "B",
    28: "D",
    29: "D",
    30: "A",
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

# Leaves indexes (bottom to top, left to right)
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




















