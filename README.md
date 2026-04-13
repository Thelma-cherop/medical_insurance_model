🏥 Medical Insurance Cost Estimator
An end-to-end Machine Learning web app that predicts annual medical insurance charges based on personal health and demographic information.
Built as part of a daily ML streak — Day 1 through Day 5.
🌐 Live Demo
https://salaryanalysis-model-lpqucprn5vqxangrhsrkur.streamlit.app/
🎯 What It Does

Input your age, sex, BMI, number of children, smoker status and region
Get an instant estimate of your annual medical insurance charges
View your personal risk profile — smoking risk, BMI category, and age group

 Tech Stack
 Python
 pandas&numpy
 scikit-learn
 Matplotlib & Seaborn
 joblibStreamlit
🧠 ML Pipeline — What Was Done
Day 1 — Exploratory Data Analysis

Detected and removed 1,435 duplicate rows (51.77% of raw data) — data was corrupted
Analyzed charge distribution — confirmed heavy right skew (min $1,121, max $63,770)
Computed correlation: age (0.298), bmi (0.198), children (0.067) — all weak
Grouped by smoker status: smokers pay 3.8x more on average ($32,050 vs $8,440)
Concluded smoker is the dominant driver before any modelling

Day 2 — Feature Engineering and Encoding

Label encoded binary columns: sex (female=0, male=1), smoker (no=0, yes=1)
One-Hot encoded region with drop_first=True to avoid dummy variable trap → 3 region columns
Log-transformed charges using np.log() to normalize the right-skewed target
Final shape after preprocessing: (1337, 9)

Day 3 — Model Training and Comparison

Trained Linear Regression and Random Forest (100 trees) on same data
Evaluated in both log scale and dollar scale
Random Forest outperformed on every metric
Actual vs Predicted plots revealed both models struggle at high charges — complex smoker interactions

Day 4 — Feature Importance Analysis

Extracted Random Forest feature importances
Smoker alone accounts for 42.31% of model's predictive power
Top 3 features: smoker (0.42), age (0.19), bmi (0.14)
Region columns ranked last — geography affects cost delivery, not health risk

Day 5 — Deployment and Bug Fixing

Serialized Random Forest model with joblib
Built Streamlit web app with 6 inputs and risk profile display
Caught and fixed a smoker encoding bug — training and app encoding were flipped
Confirmed fix with sanity checks:

Age=60, smoker=yes, BMI=35 → $46,108 ✅
Age=18, smoker=no, BMI=22 → $1,299 ✅
Age=45, smoker=no, BMI=38 → $8,309 ✅ 
model performance
r2 score= 0.8962
RMSE=$4367

├── app.py                   # Streamlit web app
├── code.ipynb               # Full ML pipeline and analysis notebook
├── insurance_model.pkl      # Trained and serialized Random Forest model
├── medical_insurance.csv    # Dataset (1,337 records after cleaning)
├── requirements.txt         # Python dependencies
└── README.md                # Project documentation
💡 Key Learnings

-Always check for duplicates before any analysis — 51% of this dataset was duplicated
-Weak numeric correlations don't mean a weak model — the signal was hiding in categorical columns
-Log transformation is essential for right-skewed targets
-Random Forest handles feature interactions that Linear Regression cannot
-Training encoding and app encoding must match exactly — a flipped mapping gives confidently wrong predictions
-Sanity checking predictions against domain knowledge catches bugs that metrics miss
