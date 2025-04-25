# Probability in Medical Diagnostics and Sleep Analysis

### **Assignment Overview**
This assignment is designed to help you apply probability concepts to analyze real-world data on sleep health and lifestyle. You'll explore various probability distributions, Bayesian inference, and hypothesis testing, using Python libraries like NumPy, Pandas, Matplotlib, and SciPy. The provided dataset contains information on individuals' sleep patterns, stress levels, physical activity, and more.

[Google Colab](https://colab.research.google.com/drive/1sFMhEcBg1HB7qZdGu9lJDnYoGJphqIQ8?usp=sharing)
[Dataset](https://www.kaggle.com/datasets/uom190346a/sleep-health-and-lifestyle-dataset?resource=download)
---

### **Dataset Description**
The dataset includes the following columns:
- `Person ID`, `Gender`, `Age`, `Occupation`, `Sleep Duration`, `Quality of Sleep`, `Physical Activity Level`, `Stress Level`, `BMI Category`, `Blood Pressure`, `Heart Rate`, `Daily Steps`, `Sleep Disorder`.

---

### **Learning Objectives**
By completing this assignment, you will:
1. Apply probability distributions (Bernoulli, Binomial, Geometric, Hypergeometric, Poisson).
2. Conduct Bayesian inference to update probabilities with new evidence.
3. Perform hypothesis testing using statistical methods.
4. Gain practical experience with data analysis using Python.

---

### **Assignment Tasks**

#### **Part 1: Dataset Exploration (Code Provided)**
Use the provided template in Google Colab to:
- Load the dataset using Pandas.
- Inspect and perform basic data cleaning if necessary.

Example snippet:
```python
import pandas as pd

# Load dataset
df = pd.read_csv('your_dataset.csv')

# Inspect dataset
print(df.head())
print(df.info())
```

---

#### **Part 2: Probability Distributions (You Implement)**

**Task 1: Bernoulli & Binomial Models**
- Define binary outcomes for good sleep quality (Quality ≥8).
- Conduct Bernoulli trials and apply the Binomial distribution.

**Task 2: Geometric Distribution**
- Model trials until the first occurrence of poor sleep quality (Quality ≤4).

**Task 3: Hypergeometric Distribution**
- Model sampling from subsets (e.g., high stress or physical activity).

**Task 4: Poisson Distribution**
- Simulate weekly occurrences of poor sleep quality.

---

#### **Part 3: Bayesian Inference (You Implement)**
- Update probability estimates of "Good Sleep Quality" given evidence (e.g., low stress).

Example Bayesian snippet:
```python
def bayes(prior, likelihood, evidence):
    return (likelihood * prior) / evidence
```

---

#### **Part 4: Hypothesis Testing (You Implement)**
- Perform statistical tests (e.g., t-tests) to compare sleep quality across different groups (e.g., high vs low stress).

Example snippet:
```python
from scipy.stats import ttest_ind

group1 = df[df['Stress Level'] <= 3]['Quality of Sleep']
group2 = df[df['Stress Level'] >= 7]['Quality of Sleep']
t_stat, p_val = ttest_ind(group1, group2)
```

---

#### **Part 5: Visualization (Code Provided)**
Use provided visualization code snippets to support your analysis.

Example snippet:
```python
import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(8, 6))
sns.histplot(df['Sleep Duration'], bins=10, kde=True)
plt.title('Sleep Duration Distribution')
plt.show()
```

---

### **Submission Guidelines**
- Submit a clearly documented Google Colab notebook (.ipynb).
- Include markdown explanations and visualizations.
- Interpret your results clearly.

---

### **Grading Criteria**
- Probability Models Implementation (30%)
- Bayesian Calculations (20%)
- Hypothesis Testing (20%)
- Visualizations (20%)
- Documentation and Clarity (10%)

Good luck!
