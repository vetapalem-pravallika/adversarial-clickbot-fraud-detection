# Adversarial Click-Bot Fraud Detection

🚀 AI-driven fraud detection system simulating a digital ads ecosystem to detect click fraud, impression fraud, and coordinated bot networks.

---

## Project Overview

This project builds an **adversarial click-bot detection system** using:

- Feature engineering (ClickTime, BidValue, Account & Campaign Balances, Click Frequency)
- Machine Learning models: **XGBoost** for fraud classification
- Risk scoring and tier assignment (Low Risk / Review / High Risk)
- SHAP explainability for feature importance
- Adversarial testing: simulating reduced bid values to test model robustness

The system is designed to simulate real-world ads fraud scenarios, reverse-engineer bot behavior, and improve detection metrics.

---

## Features

- Fraud Risk Score prediction per click
- Dataset-level risk analysis
- SHAP-based feature importance visualization
- Adversarial click-bot testing

---

## Getting Started

1. Clone the repo:
```bash
git clone https://github.com/vetapalem-pravallika/adversarial-clickbot-fraud-detection.git
cd adversarial-clickbot-fraud-detection
