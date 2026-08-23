# Insurance Claim Amount Prediction — Deployable Project

Train in Google Colab → deploy a live "Claim Compass" app on Streamlit
Community Cloud. This predicts a **continuous claim amount** (regression),
not a yes/no classification.

## Files

| File | Purpose | Push to GitHub? |
|---|---|---|
| `app.py` | Streamlit app — single & batch claim estimates, full custom UI | ✅ Yes |
| `requirements.txt` | Python dependencies | ✅ Yes |
| `.streamlit/config.toml` | Theme colors (deep navy + sky-blue accent) | ✅ Yes — keep the `.streamlit` folder name exactly |
| `claim_model.pkl` | Trained, tuned Random Forest regressor | ✅ Yes |
| `gender_encoder.pkl`, `diabetic_encoder.pkl`, `smoker_encoder.pkl`, `region_encoder.pkl` | Fitted `LabelEncoder`s for each categorical column | ✅ Yes |
| `feature_columns.pkl` | Exact column order the model expects | ✅ Yes |
| `Insurance_Claim_Prediction_Colab.ipynb` | Training notebook | Optional — nice to keep for reference |
| `insurance.csv` | Raw dataset | ❌ No — not needed at inference time |

## Step 1 — Train the model in Colab

1. Open `Insurance_Claim_Prediction_Colab.ipynb` in Google Colab.
2. Run all cells top to bottom.
3. Upload `insurance.csv` when prompted.
4. At the end, 6 files auto-download:
   - `claim_model.pkl`
   - `gender_encoder.pkl`
   - `diabetic_encoder.pkl`
   - `smoker_encoder.pkl`
   - `region_encoder.pkl`
   - `feature_columns.pkl`

## Step 2 — Arrange your project folder

```
claim-app/
├── app.py
├── requirements.txt
├── .streamlit/
│   └── config.toml
├── claim_model.pkl
├── gender_encoder.pkl
├── diabetic_encoder.pkl
├── smoker_encoder.pkl
├── region_encoder.pkl
└── feature_columns.pkl
```

## Step 3 — Test locally

```bash
cd claim-app

python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

pip install -r requirements.txt
streamlit run app.py
```

Opens at `http://localhost:8501`.

## Step 4 — Push to GitHub

```bash
git init
git add app.py requirements.txt .streamlit *.pkl
git commit -m "Insurance claim prediction app"
git branch -M main
git remote add origin https://github.com/<your-username>/<your-repo>.git
git push -u origin main
```

## Step 5 — Deploy on Streamlit Community Cloud

1. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub.
2. Click **New app**.
3. Select your repo, branch `main`, main file `app.py`.
4. Click **Deploy**.

You'll get a public URL like `https://<your-app>.streamlit.app`.

## About the dataset

`insurance.csv` has columns: `Id, age, gender, bmi, bloodpressure, diabetic,
children, smoker, region, claim`. The target is `claim` (a dollar amount),
so this is trained as a **regression** problem — the notebook compares
Linear Regression, Random Forest, and Gradient Boosting, then tunes the
best one with `GridSearchCV` before saving it.

> Note: your original notebook's code (`Code.ipynb`) was written for a
> different dataset (`train_SJC.csv`, with columns like `ClaimNumber`,
> `WeeklyWages`, `UltimateIncurredClaimCost`). Since that file wasn't the
> one you uploaded, this notebook was rebuilt from scratch to match the
> actual `insurance.csv` schema you provided.
