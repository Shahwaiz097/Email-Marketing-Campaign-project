# Uplift Modelling for Targeted Email Marketing

## Project Overview

This project uses the Hillstrom Email Analytics dataset to answer a business question:

**Which customers should receive a marketing email because the email actually increases their chance of converting?**

A normal machine learning model predicts who is likely to buy. An uplift model predicts who is more likely to buy because they received the marketing campaign.

## Dataset

The dataset contains 64,000 customers with:

- customer history features
- campaign segment: `Mens E-Mail`, `Womens E-Mail`, or `No E-Mail`
- visit outcome
- conversion outcome
- spend outcome

## Method

This project uses a simple T-Learner uplift modelling approach:

1. Load and clean the dataset
2. Create a treatment variable
3. Compare treatment and control conversion rates
4. Train one model for treated customers
5. Train one model for control customers
6. Predict conversion probability under treatment and control
7. Calculate predicted uplift
8. Rank customers by predicted uplift
9. Select the top 20% customers to target
10. Save charts, tables, and model files

## Project Structure

```text
Uplift-Modelling-Email-Marketing/
├── data/
│   ├── raw/
│   │   └── hillstrom_email_campaign.csv
│   └── processed/
│       └── customers_scored_by_predicted_uplift.csv
├── models/
│   ├── treated_model.pkl
│   └── control_model.pkl
├── notebooks/
│   └── original_notebook.ipynb
├── outputs/
│   ├── figures/
│   └── tables/
├── src/
│   └── run_uplift_project.py
├── README.md
└── requirements.txt
```

## Key Results

- The treated group had a higher conversion rate than the control group.
- Customers were ranked by predicted uplift.
- The top 20% highest-uplift customers were recommended for targeting.

## How to Run

Install the required libraries:

```bash
pip install -r requirements.txt
```

Run the project:

```bash
python src/run_uplift_project.py
```

The results will be saved in:

```text
outputs/figures/
outputs/tables/
data/processed/
models/
```

## Portfolio Summary

Built an uplift modelling project for targeted email marketing using a randomized treatment/control dataset. The project estimates incremental customer response, trains a T-Learner model, ranks customers by predicted uplift, and recommends a targeted marketing strategy based on incremental conversion rather than simple purchase probability.
