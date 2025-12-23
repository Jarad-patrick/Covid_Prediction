Project Summary

COVID-19 Classification

This project aimed to predict whether a patient has COVID-19 using demographic and symptom-based features such as age, fever, cough, gender, and city. A full supervised machine learning pipeline was implemented, including preprocessing, model training, evaluation, and decision threshold tuning.

A Logistic Regression model was first trained as a baseline, followed by a Random Forest classifier to capture non-linear relationships. While Random Forest marginally improved performance, overall metrics plateaued, indicating limited predictive signal in the available features.

Given the medical context, recall was prioritized over accuracy to reduce false negatives. By lowering the classification threshold from the default 0.5 to 0.3, recall for COVID-positive cases increased to approximately 71%, significantly reducing missed infections. This improvement came at the cost of lower precision and overall accuracy, which is an acceptable trade-off in screening scenarios.

Conclusion:
Model performance was constrained primarily by feature quality rather than model choice. The project demonstrates correct use of preprocessing pipelines, model evaluation, confusion-matrix analysis, and threshold tuning, along with sound domain-driven decision making.

Future Improvements:

Richer symptom data (severity, duration, exposure history)

Feature engineering

Cost-sensitive learning or recall-optimized objectives
