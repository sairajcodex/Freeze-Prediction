Bank Account Freeze Prediction System
Overview
This project focuses on predicting whether a bank account should be allowed, monitored, or reviewed based on transaction behavior and KYC details. The idea is to help identify risky accounts early and reduce incorrect account freezes.

Problem
Banks often face issues where:
Suspicious activity is detected too late
Genuine users get blocked due to false alarms
KYC data is not fully utilized in decision-making
So I tried to combine both transaction data and KYC information to make better predictions.

What I Built
Used transaction data like amount, frequency, and behavior patterns
Extracted KYC details using OCR (Tesseract + OpenCV)
Combined both types of data into a single ML pipeline
Built a classification model that outputs:
                                           Allow
                                           Monitor
                                           Review
To improve accuracy, I used two approaches:
                                        Weighted combination of models
                                        Stacking technique
Results
The model performs well in separating risky and safe accounts.
Using ensemble methods helped improve stability and reduce wrong predictions.                                    
