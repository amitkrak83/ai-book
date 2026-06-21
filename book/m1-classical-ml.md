# Module 1 — Classical Machine Learning *(1950s – early 2000s)*

Picture a team of engineers at a bank in 1995. Their job: build a system to detect fraudulent credit card transactions. They sit in a conference room with stacks of paper — thousands of fraud cases from the past year — and start writing rules.

"If the transaction is over $500 and the cardholder is in a different country than usual, flag it."

They add that rule to the system. Then they find a case it misses. They add another rule. Then a fraudster figures out their system and adapts. New rules. Edge cases multiply. Six months later they have 847 rules, the system is a maintenance nightmare, and fraud is still slipping through.

This is the fundamental problem that Machine Learning was invented to solve. Instead of humans writing rules, you show the machine thousands of labeled examples — "this transaction was fraud, this one wasn't" — and let it figure out the rules itself. The machine finds patterns a human would never think to look for: the unusual combination of merchant category, transaction time, and card-present flag that together signal fraud. Rules that are too subtle to write by hand, but clearly visible in the data.

That insight — *let the machine infer the rules from examples* — is the entire foundation of classical ML.

---

## Supervised vs Unsupervised vs Reinforcement Learning

Not every ML problem has the same structure. Before choosing an algorithm, you have to recognize which type of learning situation you're in. Getting this wrong means picking the completely wrong tool.

Think of three different ways a child could learn to sort fruit. In the first scenario, a parent holds up each piece of fruit and says "apple", "orange", "banana" — the child is learning from labeled examples. In the second scenario, the parent dumps a big pile of fruit on the table and says "sort these into groups that make sense" — no labels, the child invents the categories. In the third scenario, the child plays a guessing game where they say "apple?" and get a smile or a frown in return — learning from feedback on their actions over time.

Those three scenarios are **Supervised**, **Unsupervised**, and **Reinforcement Learning**.

**Supervised Learning:** You provide labeled examples — input + correct output. The model learns to map inputs to outputs. Example: 10,000 emails labeled spam/not-spam → learn to classify new emails. The word "supervised" refers to the labels — a human supervisor already decided what the right answer is for each training example.

**Unsupervised Learning:** You provide inputs with no labels. The model finds structure on its own. Example: 10,000 customer purchase histories → find natural customer segments. Nobody told the model what the segments should be. It discovers them.

**Reinforcement Learning:** An agent takes actions in an environment, receives rewards or penalties, and learns which actions maximize reward over time. Example: a game-playing agent learns chess by playing millions of games and getting +1 for wins, -1 for losses. No labeled moves — just feedback on outcomes.

**How to recognize which one your problem needs:**
- Do you have labeled outputs? → Supervised
- Do you have data but no labels, and you want to find patterns? → Unsupervised
- Is there an agent making sequential decisions with feedback? → Reinforcement

This entire module focuses on Supervised Learning (the majority of classical ML algorithms) and one key Unsupervised algorithm (K-Means). Reinforcement Learning gets its full treatment in Module 8.

---

## Foundational Concepts

Before any algorithm, there are four ideas you need to understand completely. They apply to every model you will ever train.

### Train/Test Split & Cross-Validation

Here's a tempting shortcut: train a model on your data, then test it on the same data. It scores 98%. Ship it.

Six months later, it fails completely in production. What happened?

The model memorized your training examples. It didn't learn the underlying pattern — it learned *those specific examples*. When new data arrived that it had never seen, it had nothing. This is like a student who memorizes the practice exam word for word and then faces a different exam on the same topic. They fail not because they don't understand the material, but because they never actually learned it — they just learned the practice answers.

The fix is the **train/test split**: before you touch your data, carve off 20% and lock it away. The model never sees this during training. You use it only once, at the very end, to measure true generalization performance.

```python
from sklearn.model_selection import train_test_split
import numpy as np

X = np.random.rand(1000, 5)  # 1000 examples, 5 features
y = np.random.randint(0, 2, 1000)  # binary labels

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
# 800 examples for training, 200 held out for final evaluation
print(X_train.shape, X_test.shape)  # (800, 5) (200, 5)
```

#### Cross-Validation: K-Fold vs. Stratified K-Fold
Instead of a single arbitrary split, we use **K-Fold Cross-Validation** to get a more robust estimate. We split the data into $k$ equal segments ("folds"). We train the model $k$ times: each time, $k-1$ folds are used for training and the remaining $1$ fold is held out for testing. The final performance is the average of these $k$ scores.

*   **Standard K-Fold:** Splitting is done randomly. This is fine for regression or balanced datasets.
*   **Stratified K-Fold:** Splitting is done such that each fold contains approximately the same percentage of samples of each target class as the complete set. If your dataset has 90% healthy cases and 10% cancer cases, every fold will maintain this 9:1 ratio. This is critical for classification tasks, especially with imbalanced data.

```python
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression

model = LogisticRegression()
# Stratified 5-Fold is default for classification in cross_val_score when y is categorical
scores = cross_val_score(model, X, y, cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42))
print(f"CV scores: {scores.round(3)}")
print(f"Mean: {scores.mean():.3f} ± {scores.std():.3f}")
```

#### Data Leakage: The Silent Model Killer
**Data Leakage** occurs when information from outside the training dataset is used to train the model, leading to overly optimistic training/validation scores and terrible production performance.

A classic example of data leakage is **improper preprocessing** (like scaling or imputing missing values) before splitting:
```python
# --- BAD: DATA LEAKAGE ---
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X) # Fitting on the entire dataset! Information from test set leaks here.
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2)

# --- GOOD: NO LEAKAGE ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train) # Fit ONLY on train
X_test_scaled = scaler.transform(X_test)       # Transform test using train's mean and std
```
> ⚠️ **Rule of Thumb:** Any preprocessing step that aggregates statistics (mean, variance, min, max, word frequencies) must be fit *only* on the training set. To enforce this, use Scikit-Learn `Pipeline`.

---

### Overfitting vs Underfitting

These are the two fundamental failure modes in ML. Every model you train will lean toward one of them.

Think of two students preparing for an exam. The first student reads the textbook once and understands the general concepts but doesn't practice problems. They do okay but miss nuances — **underfitting**. The second student photocopies the practice exam and memorizes every answer word for word. They score 100% on the practice exam but bomb the real one because the questions are slightly different — **overfitting**.

Your model is that second student whenever it memorizes training data quirks instead of learning the real pattern.

**Overfitting:** High training accuracy, low test accuracy. The model is too complex relative to the amount of data — it fits the noise. A decision tree with unlimited depth will memorize every training example perfectly, achieving 100% training accuracy while failing badly on new data.

**Underfitting:** Low training accuracy and low test accuracy. The model is too simple to capture the real pattern. A linear model trying to fit a curved relationship is always underfitting.

```
Training accuracy  Test accuracy  Diagnosis
     99%               62%        Overfitting — memorized training data
     61%               59%        Underfitting — too simple
     93%               90%        Good fit — generalizes well
```

**Fixing overfitting:** get more training data, use a simpler model, add regularization (covered in Module 2), use cross-validation to detect it early.

**Fixing underfitting:** use a more complex model, add more informative features, remove constraints like max_depth in decision trees.

---

### Bias–Variance Tradeoff

Overfitting and underfitting don't just happen randomly — they follow a mathematical pattern called the bias-variance tradeoff. Understanding this tells you *why* they happen and why you can't eliminate both simultaneously.

Imagine you're throwing darts at a target. **Bias** is how far your average throw is from the bullseye — systematic error from consistently aiming wrong. **Variance** is how spread out your throws are — whether you're consistent or all over the board.

A simple model is like a dart thrower who always aims slightly to the left. Every throw lands in roughly the same spot — low variance — but it's always wrong in the same direction. High bias, low variance. **Underfitting.**

A complex model is like a dart thrower who compensates differently for every throw based on random wind guesses. Sometimes they hit the bullseye, sometimes they're way off. Low bias on average but high variance — wildly inconsistent. **Overfitting.**

```
Simple model  →  High bias, low variance  →  Underfits (consistently wrong)
Complex model →  Low bias, high variance  →  Overfits (inconsistently right)
```

The total error in a model = Bias² + Variance + irreducible noise. You can't drive both bias and variance to zero simultaneously. Adding model complexity reduces bias but increases variance. The art of ML is finding the complexity that minimizes their sum.

This is why regularization (Module 2) is so important: it's the main tool for controlling variance without sacrificing too much bias.

---

### Evaluation Metrics

**Why accuracy alone is dangerous:** Suppose you build a cancer detection model. Your test set has 990 healthy people and 10 cancer patients. A model that predicts "healthy" for everyone, never detecting a single cancer case, scores 99% accuracy. That model is completely useless — and yet by accuracy alone it looks excellent.

This is the class imbalance problem, and it appears everywhere: fraud detection (0.1% of transactions are fraud), disease screening, equipment failure prediction. You need metrics that measure the things that actually matter.

**The Confusion Matrix** is the foundation. It breaks predictions into four types:

```
                    ACTUAL
                    Positive    Negative
PREDICTED  Positive    TP          FP      ← model said yes
           Negative    FN          TN      ← model said no
```

- **TP (True Positive):** Model said cancer, actually cancer. Caught it. ✅
- **FP (False Positive):** Model said cancer, actually healthy. False alarm. ❌
- **FN (False Negative):** Model said healthy, actually cancer. Missed it. Dangerous. ❌
- **TN (True Negative):** Model said healthy, actually healthy. Correct. ✅

From this matrix, four key metrics:

**Precision** — when you said "yes", how often were you right?
`Precision = TP / (TP + FP)`
High precision = few false alarms. Use when false positives are costly (spam filter — you don't want real emails blocked).

**Recall** — of all actual positives, how many did you catch?
`Recall = TP / (TP + FN)`
High recall = few missed cases. Use when false negatives are costly (cancer detection — you never want to miss a case).

**F1 Score** — harmonic mean of precision and recall. Useful when you want one number that penalizes extreme imbalance.
`F1 = 2 × (Precision × Recall) / (Precision + Recall)`

**AUC-ROC** — plots true positive rate vs false positive rate at every possible decision threshold. AUC=1.0 is perfect, AUC=0.5 is random. Threshold-independent — tells you about overall discriminative ability.

```python
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

y_true  = [1, 0, 1, 1, 0, 1, 0, 0, 1, 0]
y_pred  = [1, 0, 0, 1, 0, 1, 1, 0, 1, 0]
y_proba = [0.9, 0.2, 0.4, 0.8, 0.1, 0.95, 0.6, 0.15, 0.85, 0.3]

print(confusion_matrix(y_true, y_pred))
print(classification_report(y_true, y_pred))
print(f"AUC: {roc_auc_score(y_true, y_proba):.3f}")
```

**The Precision–Recall tradeoff:** you can always increase precision by being more selective (only flag cases you're 99% sure about), but recall will drop as you miss more borderline cases. You can always increase recall by flagging everything, but precision collapses. The right balance depends on your application.

| Application | Worse failure | Optimize for |
|-------------|---------------|--------------|
| Cancer detection | Missing cancer (FN) | Recall |
| Spam filter | Blocking real emails (FP) | Precision |
| Fraud detection | Missing fraud (FN) | Recall |
| Content moderation | Over-removing valid content (FP) | Precision |

**Regression metrics** apply when your model outputs a number rather than a category:

- **MAE (Mean Absolute Error):** Average absolute difference between prediction and actual. Easy to interpret — same units as output. Robust to outliers.
- **RMSE (Root Mean Squared Error):** Like MAE but squares errors first, so large errors count much more. Use when big mistakes are disproportionately bad.
- **R² (R-squared):** Proportion of variance in the output your model explains. R²=1 is perfect; R²=0 means the model is no better than predicting the mean; R²<0 means it's actively worse than the mean.

```python
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

y_true = [200000, 350000, 150000, 450000]
y_pred = [210000, 330000, 160000, 420000]

print(f"MAE:  ${mean_absolute_error(y_true, y_pred):,.0f}")
print(f"RMSE: ${np.sqrt(mean_squared_error(y_true, y_pred)):,.0f}")
print(f"R²:   {r2_score(y_true, y_pred):.3f}")
```

> Domain-specific metrics covered in their respective modules: CV metrics (IoU, mAP, mIoU) in Module 3 · NLP metrics (BLEU, ROUGE) in Module 4 · LLM & Ranking metrics (Perplexity, NDCG) in Module 6.

---

### Imbalanced Data & SMOTE

In many real-world problems (fraud detection, rare diseases, spam), one class is far more common than the other. Accuracy is a dangerous metric here because a naive model predicting the majority class gets 99% accuracy on a dataset with 1% minority samples.

To address class imbalance, we use two main strategies:
1.  **Metric Selection:** Focus on Precision, Recall, F1-Score, or Precision-Recall AUC (PR-AUC) rather than Accuracy.
2.  **Resampling the Data:**
    *   *Under-sampling:* Randomly remove majority class samples (loses valuable information).
    *   *Over-sampling:* Duplicate minority class samples (leads to overfitting).
    *   **SMOTE (Synthetic Minority Over-sampling Technique):** Instead of duplicating samples, SMOTE creates *synthetic* examples of the minority class. It does this by drawing lines between existing minority samples that are close to each other in the feature space and placing new points along those lines.

```python
# Synthesize data for demonstration
# Requires installing 'imbalanced-learn' (pip install imbalanced-learn)
# SMOTE is always applied ONLY to the training dataset, never the test dataset!

from imblearn.over_sampling import SMOTE
from collections import Counter

print(f"Original training class distribution: {Counter(y_train)}")
# If unbalanced: Counter({0: 720, 1: 80})

smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

print(f"Resampled training class distribution: {Counter(y_train_resampled)}")
# Counter({0: 720, 1: 720}) - perfectly balanced
```

---

### Feature Engineering & Feature Selection

Raw data is rarely ready for ML. **Feature Engineering** is the process of creating new features or transforming existing ones to make it easier for algorithms to learn patterns.
*   *Domain Transformations:* Turning timestamp into "hour of day", "is_weekend".
*   *Interaction Features:* Multiplying house size by number of bedrooms.
*   *Encoding:* One-Hot Encoding categorical variables.

Once you have many features, you must perform **Feature Selection** to remove redundant, noisy, or irrelevant inputs. This reduces overfitting, speeds up training, and makes models easier to interpret.

There are three main categories of feature selection:

1.  **Filter Methods:**
    *   **How they work:** Evaluate each feature independently of the model using statistical tests (e.g., correlation, Chi-Square, Mutual Information). Features with low scores are dropped.
    *   **Pros/Cons:** Very fast, but ignores feature interactions (two features might be weak individually but strong together).
2.  **Wrapper Methods:**
    *   **How they work:** Use a model as a search engine. Train the model on different subsets of features and evaluate. Examples: Forward Selection (start with 0 features and add one by one), Backward Elimination (start with all features and prune).
    *   **Pros/Cons:** Finds the best performing combination, but computationally expensive (requires retraining models hundreds of times).
3.  **Embedded Methods:**
    *   **How they work:** Feature selection occurs naturally during model training.
    *   **Examples:** L1 Regularization (Lasso) which forces weights of unimportant features to 0; Decision Trees/Random Forests which score features based on how often they are used for splits.
    *   **Pros/Cons:** Computational efficiency of filters with the model accuracy of wrappers.

---

### Ensembling: Voting, Bagging, Boosting, Stacking

If one model makes a prediction, it might have high bias or high variance. **Ensemble methods** combine the predictions of multiple models to produce a single, more robust prediction. There are four main ensembling paradigms:

```
                  ┌────────────────── ENSEMBLING ──────────────────┐
                  │                                                │
           ┌──────┴──────┐                                  ┌──────┴──────┐
       Parallel         Seq                              Meta-Model      Simple
       (Bagging)     (Boosting)                          (Stacking)     (Voting)
   e.g. Random Forest  e.g. XGBoost                        Combine       Average or
                                                          predictions    majority vote
```

1.  **Voting:**
    *   **How it works:** Train several different models (e.g. Logistic Regression, SVM, KNN) on the same dataset. For predictions, take the majority vote (classification) or average the outputs (regression).
2.  **Bagging (Bootstrap Aggregating):**
    *   **How it works:** Train multiple instances of the *same* algorithm (usually Decision Trees) in parallel. Each instance is trained on a random sample of the data (with replacement). Predictions are averaged.
    *   **Primary goal:** Reduces model **variance** (overfitting). Random Forest is the classic example.
3.  **Boosting:**
    *   **How it works:** Train models sequentially rather than in parallel. Each new model is trained to correct the errors made by the previous models.
    *   **Primary goal:** Reduces model **bias** (underfitting). Examples: AdaBoost, Gradient Boosting.
4.  **Stacking (Stacked Generalization):**
    *   **How it works:** Train multiple base models (e.g. Random Forest, SVM). Take their predictions and use them as inputs to train a *meta-model* (usually a simple Logistic Regression) which learns how to best combine their outputs.

---

## Supervised Algorithms

### Linear Regression *(1805 origin, ML use from 1950s–60s)*

The very first question anyone ever wanted to answer with data was: "given what I know, what number should I expect?" A farmer wanting to predict crop yield from rainfall. An engineer wanting to predict metal fatigue from stress cycles. A banker wanting to predict loan repayment from income.

All of these are asking for a number — not a category. And the simplest model for predicting a number from inputs is a straight line.

**What it is:** Linear Regression fits a straight line (or hyperplane in multiple dimensions) through data. It predicts an output `y` as a weighted sum of inputs: `y = w₁x₁ + w₂x₂ + ... + b`. The weights `w` and bias `b` are learned from data by minimizing the Mean Squared Error (MSE) — the average of squared differences between predictions and actual values.

`MSE = (1/n) Σ (yᵢ - ŷᵢ)²`

This is solved analytically (the Normal Equation gives an exact solution) or via gradient descent for large datasets.

```python
from sklearn.linear_model import LinearRegression
import numpy as np

# Predict house price from size (sq ft)
X = np.array([[800], [1200], [1500], [1800], [2200]])
y = np.array([200000, 280000, 340000, 400000, 480000])

model = LinearRegression()
model.fit(X, y)

print(f"Price per sq ft: ${model.coef_[0]:.0f}")    # learned weight
print(f"Base price: ${model.intercept_:,.0f}")        # learned bias
print(f"Predict 1600 sqft: ${model.predict([[1600]])[0]:,.0f}")
```

**Where it's used:** House price prediction, demand forecasting, financial modeling, A/B test analysis, medical dosage prediction, energy consumption forecasting. Linear Regression is the baseline model in virtually every regression problem — you always start here before trying anything more complex.

**Failure modes:** Assumes a linear relationship between inputs and output. Real-world relationships are often curved, seasonal, or driven by interaction effects — none of which a straight line can capture. Sensitive to outliers (a single wildly wrong data point pulls the line toward it). Also breaks down when input features are highly correlated with each other (multicollinearity makes weight estimates unstable).

---

### Logistic Regression *(1958)*

The fraud analysts at our bank eventually got frustrated with rules. Their manager had a different idea: "what if we train a model that outputs a *probability* of fraud for each transaction?" A score of 0.92 means 92% likely fraud. A score of 0.03 means probably fine. Then a human or automated system decides what to do based on the score.

The problem: Linear Regression outputs numbers like 1.7 or -0.3 — which can't be probabilities. They needed something that always stays between 0 and 1.

**What it is:** Logistic Regression takes the output of a linear equation and squashes it through a **sigmoid function** that maps any real number to the range (0, 1).

`σ(z) = 1 / (1 + e^(-z))`

Despite the name, it's a **classifier**, not a regressor. The output is a probability. A threshold (usually 0.5) converts it to a binary prediction. Training minimizes **binary cross-entropy loss**, which heavily penalizes confident wrong predictions.

```python
from sklearn.linear_model import LogisticRegression
import numpy as np

# Predict pass/fail from hours studied
X = np.array([[1], [2], [3], [5], [6], [7], [8]])
y = np.array([0, 0, 0, 1, 1, 1, 1])  # 0=fail, 1=pass

model = LogisticRegression()
model.fit(X, y)

prob = model.predict_proba([[4]])[0][1]
print(f"4 hours studied → {prob:.1%} probability of passing")
```

**Where it's used:** Credit scoring (probability of default), fraud detection, medical diagnosis (probability of disease), email spam filtering, click-through rate prediction in ad systems, customer churn prediction. It's the most widely deployed ML algorithm in the financial industry because it's fast, interpretable, and gives calibrated probabilities.

**Failure modes:** Can only learn a linear decision boundary — a straight line (or hyperplane) separating classes. If the true boundary is curved or complex, Logistic Regression underfits. Requires feature scaling (otherwise large-magnitude features dominate). Also struggles when features are highly correlated with each other.

---

### Decision Tree *(1960s–80s, ID3 formalized 1986)*

The fraud analysts found that Logistic Regression's linear boundary wasn't catching enough cases. The fraud patterns weren't linear — they were more like rules: "if the merchant category is electronics AND the transaction is over $1,000 AND it's between midnight and 4am, flag it." A straight line can never express that kind of compound rule.

What they needed was a machine that *learned rules* from data.

**What it is:** A Decision Tree learns a series of if-else rules by recursively splitting the data on the feature that best separates the classes. Each split creates a branch; the leaves hold final predictions.

At each node, the algorithm tests every feature and every possible split point, choosing the split that maximizes **information gain** (or minimizes **Gini impurity**) — measures of how much purer the resulting groups become. A split that puts all fraud cases on one side and all legitimate cases on the other is a perfect split.

```python
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.datasets import load_iris

X, y = load_iris(return_X_y=True)
model = DecisionTreeClassifier(max_depth=3, random_state=42)
model.fit(X, y)

# Print the actual rules the model learned — this is the interpretability superpower
print(export_text(model, feature_names=list(load_iris().feature_names)))
```

**Where it's used:** Medical diagnosis (interpretable rules doctors can verify), loan approval (regulators require explainable decisions), customer segmentation, fraud rule discovery, manufacturing quality control. Decision Trees are prized for **interpretability** — you can print exactly what rules were learned and explain them to a non-technical stakeholder. This matters enormously in regulated industries like finance and healthcare.

**Failure modes:** Unconstrained trees overfit spectacularly — given enough depth, a tree memorizes every training example. They're also unstable: small changes in training data can produce completely different trees (high variance). They can only make axis-aligned splits (horizontal or vertical lines), struggling with diagonal decision boundaries. These problems directly motivated the invention of Random Forests.

---

### Naive Bayes *(1960s)*

In the late 1990s, spam email was becoming a crisis. Inboxes were flooding with "FREE MONEY" and "WIN A PRIZE" messages. The challenge: build a filter that runs on 1990s hardware, processes thousands of emails per second, and learns from examples — not hand-written rules.

Logistic Regression needed careful feature engineering and struggled with the tens of thousands of possible word features in emails. Decision Trees were unstable on high-dimensional text data. What was needed was something that could learn from a small dataset, handle massive feature spaces, and classify fast enough to filter every incoming email in real time.

**What it is:** Naive Bayes applies **Bayes' theorem** to classification. Given a new email, it asks: "what's the probability this is spam, given the words I see?"

`P(spam | words) ∝ P(spam) × P(words | spam)`

It's "naive" because it assumes all features (words) are **independent** of each other given the class — which is obviously false ("free" and "money" tend to appear together). But this simplification works remarkably well in practice, especially for text, and makes the math tractable with very little data.

For each class, count how often each word appears in training examples. For a new email, multiply the word probabilities together and pick the class with the highest product.

```python
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer

texts = ["free money now", "hello friend", "win cash prize", "meeting at noon"]
labels = [1, 0, 1, 0]  # 1=spam, 0=not spam

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(texts)

model = MultinomialNB()
model.fit(X, labels)

test = vectorizer.transform(["free cash prize"])
print("Spam?", model.predict(test)[0])  # 1 → spam
```

**Where it's used:** Email spam filtering (its original killer application, deployed at scale at Yahoo and early Gmail), document classification, sentiment analysis, medical text classification, real-time content filtering. It was the algorithm that largely solved the spam crisis of the early 2000s.

**Failure modes:** The independence assumption breaks down when features are strongly correlated, producing poorly calibrated output probabilities (extreme values like 0.0001 or 0.9999 are common). A word that appears in training but never in a given class produces zero probability for that class — overriding all other evidence — fixed with Laplace smoothing. For tasks with abundant data and complex patterns, more expressive models outperform it.

---

### K-Nearest Neighbors *(1967)*

Sometimes you don't want to fit any model at all. You just want to ask: "what do the most similar examples in my training data look like?"

Think of a real estate agent estimating a house price. They don't run a regression — they look up the three most similar houses that sold recently in the same neighborhood and price accordingly. That intuition is exactly KNN.

**What it is:** KNN doesn't "train" in the traditional sense — it memorizes the entire training set. For a new example, it finds the K closest training examples by distance and aggregates their labels: majority vote for classification, average for regression.

Distance is usually Euclidean: `d = √(Σ(xᵢ - yᵢ)²)`, though Manhattan or cosine distance are used depending on the data type.

K is a hyperparameter you choose:
- Small K (K=1) → complex decision boundary, easily influenced by noise, high variance
- Large K → smoother boundary, but can blur real patterns, high bias

```python
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Always scale before KNN — distance is meaningless without it
model = Pipeline([('scaler', StandardScaler()), ('knn', KNeighborsClassifier(n_neighbors=5))])
model.fit(X_train, y_train)
print(f"Accuracy: {model.score(X_test, y_test):.1%}")
```

**Where it's used:** Recommendation systems ("users similar to you also liked..."), anomaly detection (a point with no close neighbors is suspicious), image retrieval (find visually similar images), medical diagnosis based on patient similarity, collaborative filtering. Also widely used as a baseline — if a fancier model can't beat KNN on your data, something is wrong with the fancier model.

**Failure modes:** Devastated by the **curse of dimensionality** — in high-dimensional spaces, every point becomes roughly equidistant from every other, so "nearest neighbors" stops being meaningful. Slow at prediction time (must compute distance to every training point for each query) and memory-hungry (stores all training data). Feature scaling is mandatory — income measured in thousands will dominate distance calculations over age measured in decades if you don't normalize.

---

### Support Vector Machines *(1995)*

By the mid-1990s, researchers had a nagging question. When two classes are linearly separable, there are infinitely many straight lines that correctly divide them. But which line is actually *best* for generalizing to new data?

The answer turned out to be elegant: the line that stays as far as possible from both classes. Maximum margin.

**What it is:** SVMs find the **maximum-margin hyperplane** — the decision boundary that maximizes the gap to the nearest training examples of each class. Those nearest examples are called **support vectors** — they're the only points that matter for defining the boundary. Everything else in the dataset is irrelevant once the support vectors are found.

The genius of SVMs is the **kernel trick**. Many problems aren't linearly separable in the original feature space — but if you project the data into a higher-dimensional space, a hyperplane might separate it there. The kernel trick computes this implicitly without ever actually building the high-dimensional representation — making it computationally feasible. Common kernels: RBF (handles curved boundaries), polynomial.

The `C` parameter controls the margin/error tradeoff: small C allows some misclassifications for a wider margin (more robust to noise), large C demands correct classification at the cost of a narrower margin (can overfit).

```python
from sklearn.svm import SVC
from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

X, y = make_classification(n_samples=200, n_features=2,
                           n_informative=2, n_redundant=0, random_state=42)

# Always scale before SVM — kernel computations depend on feature magnitude
model = Pipeline([('scaler', StandardScaler()), ('svc', SVC(kernel='rbf', C=1.0))])
model.fit(X, y)
print(f"Support vectors per class: {model.named_steps['svc'].n_support_}")
```

**Where it's used:** Text classification (SVMs dominated NLP benchmarks from 1998–2012 before deep learning took over), bioinformatics (gene classification, protein structure prediction where datasets are small but high-dimensional), image classification on small datasets, handwriting recognition. In scientific domains where data is scarce but features are many (genomics, spectroscopy), SVMs still compete with neural networks.

**Failure modes:** Training time scales roughly O(n²) to O(n³) with number of training examples — impractical for datasets with millions of rows. Don't directly output calibrated probabilities (require Platt scaling as post-processing). Kernel and hyperparameter (`C`, `gamma`) selection requires careful cross-validation. Unlike Decision Trees, they're a black box — you can't read the learned rules.

---

### Random Forest *(2001)*

The fraud team ran Decision Trees for a while. They noticed something frustrating: whenever they retrained the model on slightly different data — adding a month of new transactions — the tree changed completely. Individual predictions became inconsistent and hard to trust.

Their lead engineer had an insight borrowed from democracy: if one expert can be wrong, maybe a hundred experts voting together will be right. Even if each tree makes different mistakes, the mistakes won't all be the same mistakes — and when you average many different wrong answers, the errors cancel out.

**What it is:** Random Forest trains many Decision Trees — typically 100 to 500 — each on a **random bootstrap sample** of the training data (sampling with replacement) and using only a **random subset of features** at each split. At prediction time, all trees vote (classification) or average (regression).

The two randomness sources are crucial: bootstrap sampling ensures each tree sees a different slice of data, and feature randomness ensures trees can't all learn the same dominant pattern. This *diversity* is what makes the ensemble robust. When many independent models agree, it's likely signal; when they disagree, it's likely noise.

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
print(f"Accuracy: {model.score(X_test, y_test):.1%}")

# Feature importances: which inputs mattered most to the prediction?
for name, importance in zip(load_iris().feature_names, model.feature_importances_):
    print(f"  {name}: {importance:.3f}")
```

**Where it's used:** Credit scoring, fraud detection, medical diagnosis, drug discovery, customer churn prediction, stock screening. Random Forest consistently ranks among the top performers on tabular (structured) data benchmarks — it's typically the first non-linear model practitioners reach for after establishing a linear baseline. The feature importance scores are a bonus: a practical tool for understanding which variables drive predictions, useful for debugging and for explaining models to stakeholders.

**Failure modes:** Much less interpretable than a single Decision Tree — you can't read 100 trees to understand the logic. Slower to train and predict than single models. Doesn't extrapolate beyond the range of training data (if training prices go up to $1M, it can't reliably predict a $5M house). Doesn't naturally handle sequential or time-dependent patterns.

---

### Boosting Evolution (GBM, XGBoost, LightGBM, CatBoost) *(2000s–2010s)*

The bank's fraud analysts noticed a problem: Random Forests are built by training many independent trees in parallel and averaging them. While this is great at reducing **variance** (overfitting), it cannot improve a model that suffers from high **bias** (underfitting). If your individual decision trees are too shallow and simple to capture the pattern, averaging 100 of them just gives you the same mediocre prediction.

To solve this, researchers turned to **Boosting**: building trees *sequentially* rather than in parallel, where each tree is dedicated to correcting the specific mistakes of the trees that came before it.

#### 1. Gradient Boosting Machine (GBM)
*   **What it is:** Instead of re-weighting data points (like AdaBoost), GBM fits a new decision tree to predict the **residuals** (the errors) of the collective model so far. In mathematical terms, it performs gradient descent in the functional space, optimizing a loss function (like MSE for regression or Log Loss for classification) step-by-step.
*   **The formula:** For each step $m$, the model is updated as: $F_m(x) = F_{m-1}(x) + \gamma_m h_m(x)$, where $h_m(x)$ is the new tree predicting residuals, and $\gamma_m$ is the learning rate (or shrinkage factor) to prevent overfitting.

#### 2. XGBoost (Extreme Gradient Boosting - 2014)
*   **Why it was invented:** Standard GBM is slow (sequential training can't easily run on multiple CPU cores) and prone to overfitting. XGBoost was created by Tianqi Chen to solve this.
*   **Key Innovations:**
    *   *Regularization:* Adds L1 and L2 regularization penalties directly to the tree objective function to limit tree complexity.
    *   *Hardware Optimization:* Implements a block structure for parallel computation and cache-aware access patterns.
    *   *Handling Missing Values:* Learns a default direction for missing values during split finding.
    *   *Second-Order Gradients:* Uses Taylor expansion to approximate the loss function, allowing it to work with any custom differentiable loss function.

#### 3. LightGBM (Microsoft - 2016)
*   **Why it was invented:** On datasets with millions of rows and hundreds of features, XGBoost becomes slow because it scans every data point for every possible split.
*   **Key Innovations:**
    *   *Leaf-wise Growth:* Instead of growing trees level by level, LightGBM grows the tree by splitting the leaf that reduces loss the most (leaf-wise). This produces deeper, more accurate trees, though it requires constraints like `max_depth` to avoid overfitting.
    *   *GOSS (Gradient-based One-Side Sampling):* Retains all samples with large gradients (errors) and randomly samples those with small gradients, drastically reducing the data size checked for splits.
    *   *EFB (Exclusive Feature Bundling):* Combines mutually exclusive features (features that rarely take non-zero values simultaneously, common in sparse matrices) to reduce feature dimensions.
    *   *Speed:* Trains up to 10× faster than XGBoost with significantly lower memory usage.

#### 4. CatBoost (Yandex - 2017)
*   **Why it was invented:** Real-world tabular data is full of categorical features (e.g. zip codes, device types, browser names). Pre-processing these with One-Hot Encoding creates massive sparse matrices that slow down boosting.
*   **Key Innovations:**
    *   *Symmetric Trees:* CatBoost builds oblivious trees (symmetric trees where the same split condition is applied to all nodes at a given level). Oblivious trees are balanced, less prone to overfitting, and extremely fast at prediction time.
    *   *Ordered Boosting:* A technique to calculate target statistics for categorical features using historical orders of training data to prevent target leakage (overfitting during category encoding).

```python
# Requires: pip install xgboost lightgbm catboost
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

X, y = make_classification(n_samples=1000, n_features=20, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# XGBoost
xgb = XGBClassifier(n_estimators=100, learning_rate=0.05, max_depth=5, random_state=42)
xgb.fit(X_train, y_train)
print(f"XGBoost Test Accuracy: {xgb.score(X_test, y_test):.1%}")

# LightGBM
lgb = LGBMClassifier(n_estimators=100, learning_rate=0.05, max_depth=5, random_state=42)
lgb.fit(X_train, y_train)
print(f"LightGBM Test Accuracy: {lgb.score(X_test, y_test):.1%}")

# CatBoost
cat = CatBoostClassifier(iterations=100, learning_rate=0.05, depth=5, random_state=42, verbose=0)
cat.fit(X_train, y_train)
print(f"CatBoost Test Accuracy: {cat.score(X_test, y_test):.1%}")
```

**Where they're used:** Boosting algorithms (especially XGBoost and LightGBM) are the undisputed kings of tabular data on Kaggle. They power search engines, recommendation feeds, credit scoring systems, insurance underwriting, and click-through-rate predictions across the tech industry.

**Failure modes:** Highly sensitive to hyperparameters (learning rate, depth, regularization coefficients) requiring careful tuning. Overfits quickly if training is not stopped early (using validation sets). Slower to train than Random Forests because trees must be built sequentially.

---

## Unsupervised Algorithm

### K-Means Clustering *(1957/1967)*

A marketing team at an e-commerce company has a database of 5 million customers. They want to run targeted campaigns — different messages for different types of customers. But they have no idea what "types" of customers they have. Nobody labeled the customers. They just have purchase histories, browsing behavior, and demographics.

This is an unsupervised problem: find the natural groups hiding in data that nobody has labeled.

**What it is:** K-Means divides data into K clusters, where each cluster is defined by its **centroid** — the mean position of all points assigned to it. Every data point belongs to the cluster whose centroid is closest (by Euclidean distance). You choose K; the algorithm finds the best centroids for that K.

**How it works** — two alternating steps until convergence:
1. **Assign:** each point is assigned to the nearest centroid
2. **Update:** each centroid moves to the mean of its assigned points

This continues until assignments stop changing. It's guaranteed to converge but not guaranteed to find the global optimum (it can get stuck in local minima), which is why you run it multiple times with different random initializations and pick the best result.

```python
from sklearn.cluster import KMeans
import numpy as np

# Customer data: [monthly spend ($), purchase frequency per month]
X = np.array([[10, 1], [12, 2], [9, 1],    # low-value: infrequent, small spend
              [80, 10], [90, 12], [85, 11], # high-value: frequent, big spend
              [45, 5], [50, 6], [48, 5]])   # mid-value

model = KMeans(n_clusters=3, random_state=42, n_init=10)
model.fit(X)
print("Cluster assignments:", model.labels_)
print("Centroids:\n", model.cluster_centers_.round(1))
# The 3 centroids naturally correspond to the 3 customer tiers
```

**Choosing K:** The **elbow method** — run K-Means for K=2 through 15, plot the within-cluster sum of squares (WCSS) for each. WCSS drops as K increases (more clusters always fit better). Find the "elbow" — where adding another cluster gives diminishing returns. The silhouette score gives a more rigorous answer: it measures how well each point fits its own cluster versus the next-nearest cluster.

**Where it's used:** Customer segmentation (the original motivating application), document clustering, image compression (replacing pixel colors with cluster centroids), anomaly detection (points far from all centroids are likely outliers), recommendation system preprocessing, gene expression analysis in biology.

**Failure modes:** Assumes clusters are roughly spherical and similar in size — struggles with elongated clusters, very different cluster densities, or non-convex shapes. Sensitive to outliers (a single extreme point can pull a centroid far from the true cluster center). Sensitive to initial centroid placement (sklearn's k-means++ initialization largely mitigates this). And you must choose K in advance — if the true number of clusters isn't known, it requires exploration.

---

## Where Classical ML Breaks Down

Classical ML dominated from the 1950s through the early 2000s. It produced real value: fraud detectors, spam filters, credit scores, medical classifiers, recommendation systems. But by the mid-2000s, a ceiling became visible. The same problem kept appearing in different domains. It had a name: **feature engineering**.

Every classical ML algorithm takes a table of numbers as input — rows are examples, columns are features. For a loan application, features are obvious: income, credit score, debt-to-income ratio. A human expert can define them. For house price prediction, you can define square footage, bedrooms, neighborhood. Fine.

But what about an image? A 224×224 image is 150,528 raw pixel values. A Random Forest can't learn from raw pixels — the curse of dimensionality would destroy it. So a human expert has to extract features first: edge counts, color histograms, texture statistics. But which features? The choice is arbitrary, domain-specific, and never quite captures what the model actually needs to know. Teams of computer vision researchers spent decades carefully hand-crafting feature extractors — HOG features, SIFT descriptors, Gabor filters — and every new domain required starting over.

Text is the same story. The sentence "the movie was not boring" can't go directly into a classifier. You need features: word counts, n-grams, sentiment lexicons. But bag-of-words features treat "not boring" and "boring" as nearly identical because they share a key word. Capturing the actual meaning of a sentence required handcrafted linguistic features that NLP researchers worked on for decades without fully solving.

The fundamental bottleneck: **humans deciding which features matter**. It's slow. It requires domain expertise. It doesn't transfer between domains. And it always leaves signal on the table — patterns that exist in the raw data but that no human thought to look for.

What the field needed was a model that could look at raw data — raw pixels, raw text — and learn its own features. Not features a human designed, but features the model discovered were useful for the task. Learn the edges and textures from pixels. Learn word meanings from text sequences. Let the model find the representation, not the human.

That idea is the entire premise of deep learning. Stack many layers of simple transformations, and the early layers learn low-level patterns (edges, letter shapes) while later layers learn complex abstractions (faces, sentiment, syntax). The features emerge from the data rather than being hand-specified.

Every limitation of classical ML — the hand-engineering bottleneck, the curse of dimensionality on raw inputs, the inability to capture sequential context — has a direct answer in Module 2. The story of deep learning begins exactly where classical ML runs out of road.

---

## Quick Reference & Common Questions

### Algorithm Cheat Sheet

| Algorithm | Best for | Needs feature scaling? | Interpretable? |
|-----------|----------|----------------------|----------------|
| Linear Regression | Predicting numbers, baseline model | Yes | ✅ Yes |
| Logistic Regression | Binary/multi-class, calibrated probabilities | Yes | ✅ Yes |
| Decision Tree | Non-linear rules, explainable decisions | No | ✅ Yes |
| Naive Bayes | Text classification, small data, speed | No | Partial |
| KNN | Similarity-based tasks, baselines | Yes | ✅ Yes |
| SVM | High-dimensional, small-medium datasets | Yes | ❌ No |
| Random Forest | Tabular data, robustness, feature importance | No | ❌ No |
| K-Means | Clustering, segmentation, compression | Yes | Partial |

### Analogy Reference

| Algorithm | Simple analogy |
|-----------|---------------|
| **Linear Regression** | Drawing the best straight line through dots on a scatter plot |
| **Logistic Regression** | Linear Regression with an S-bend so output is always 0–100% |
| **Decision Tree** | Playing 20 Questions — each question eliminates half the possibilities |
| **Random Forest** | 100 people play 20 Questions on different random subsets — take the majority vote |
| **SVM** | Draw the widest possible road between two groups of dots |
| **KNN** | "Show me your 5 nearest neighbors and I'll tell you who you are" |
| **K-Means** | Sorting LEGO bricks into K buckets, re-centering each bucket, then re-sorting |
| **Naive Bayes** | Count word frequencies in spam vs real emails, multiply the probabilities |

---

### Module 1 Q&A

**Q: What's the difference between supervised and unsupervised learning?**
A: Supervised = you provide labels (answers) for every training example. The model learns to predict those labels. Unsupervised = no labels — the model finds structure on its own. Email spam detection is supervised (you labeled emails as spam/not-spam). Customer segmentation is unsupervised (you don't know what segments exist — the algorithm discovers them).

**Q: Why does overfitting happen?**
A: The model has too much capacity relative to the amount of training data, so it memorizes noise and quirks specific to training examples instead of learning the underlying pattern. A decision tree with no depth limit achieves 100% training accuracy by memorizing every example — but fails on new data because those specific examples aren't the pattern, they're just the training data.

**Q: What is regularization and why is it needed?**
A: Regularization adds a penalty to the training loss that discourages overly complex models. L1 (Lasso) adds the sum of absolute weights — drives some weights to exactly zero, effectively doing feature selection. L2 (Ridge) adds the sum of squared weights — shrinks all weights toward zero but rarely to exactly zero. Both fight overfitting by constraining model complexity. More detail in Module 2.

**Q: When would you prefer Random Forest over a single Decision Tree?**
A: Almost always. A single Decision Tree has high variance — tiny changes in training data can produce wildly different trees. Random Forest averages hundreds of trees, canceling out individual tree errors. The trade-off: you lose the interpretability of a single readable tree, but accuracy improves substantially.

**Q: When should you use SVM vs Logistic Regression?**
A: For high-dimensional data with relatively few samples (text classification with many word features, few documents), SVM often wins because the maximum-margin criterion is more robust. For large datasets or when calibrated probabilities matter (you need "73% probability of default," not just "default/not default"), Logistic Regression is faster and more naturally calibrated.

**Q: What's the curse of dimensionality?**
A: As the number of features grows, the volume of the feature space grows exponentially. Data becomes sparse — in high dimensions, every point is roughly equidistant from every other. KNN breaks because "nearest neighbors" stops being meaningful. Classical ML algorithms applied to raw images fail for the same reason — 150,528 pixel features create a space too vast for algorithms designed for dozens of features.

**Q: How do you choose K in K-Means?**
A: The elbow method: run K-Means for K=2 through 15, plot within-cluster sum of squares (WCSS) for each K. Find the "elbow" where adding another cluster gives diminishing returns. The silhouette score is more rigorous — it measures how well each point fits its own cluster vs the next-nearest cluster. Values range from -1 to 1; higher is better.

**Q: What's cross-validation and why is it better than a single train/test split?**
A: Cross-validation (k-fold) splits data into k equal parts, trains k times — each time holding out a different part as validation — then averages the k scores. Better than a single split because every example is in the test set exactly once across k rounds, giving a more reliable estimate of true generalization performance by averaging out the luck of any single split.

**Q: My model has 95% accuracy. Is it good?**
A: Depends entirely on the baseline. If 95% of examples belong to one class, a model that always predicts that class gets 95% accuracy with zero learning. Always compare against a naive baseline (most-common-class prediction). Also check precision, recall, and F1 — accuracy alone is deeply misleading on imbalanced data.

**Q: Can Naive Bayes handle continuous features?**
A: Yes — Gaussian Naive Bayes assumes features follow a normal distribution and estimates mean and variance per class. MultinomialNB handles integer count features (word counts in text). BernoulliNB handles binary features (word present/absent). Choose the variant that matches your data type.
