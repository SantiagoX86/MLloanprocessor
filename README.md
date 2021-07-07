# Machine Learning Loan Processor

Presentation link: https://docs.google.com/presentation/d/1udJU0EcDkeace4no5Q5c2jofGTmUDvzCwC1U5XAU0W4/edit#slide=id.gb80c73b70a_0_42

WAU Bank has brought in our team to try to streamline their loan processing practices. While loans can take some effort and labor hours to process, automating some of that process could significantly minimize loan processing costs for the bank. I will be using a Gradient Boosting Classifer to analyze historical loan data and create a binary classification model labeling bad loans 0 and good loans 1 where bad loans are those that are at least 31 days behind on payments, have been written off, or defaulted and good loans are those less than 31 days behind on payments, in-good-standing, or paid-off. This model will enable the bank to easily flag loan applications as approved to move to the info verification phase or needing further analysis in order to be approved. I will be optimizing for specificity so as to minimize the number of approved loans that turn out to be bad loans.

### Dataset:

* Lenders Club 2007-2018 (by Nathan George) - https://www.kaggle.com/wordsforthewise/lending-club?select=accepted_2007_to_2018Q4.csv.gz

### Data Dictionary:

**Lenders Club 2007-2018**

|Feature|Description|
|--------|-------|
|loan_amnt|The listed amount of the loan applied for by the borrower. If at some point in time, the credit department reduces the loan amount, then it will be reflected in this value.|
|term|The number of payments on the loan. Values are in months and can be either 36 or 60.|
|annual_inc|The self-reported annual income provided by the borrower during registration.|
|pymnt_plan|Indicates if a payment plan has been put in place for the loan|
|dti|A ratio calculated using the borrower’s total monthly debt payments on the total debt obligations, excluding mortgage and the requested LC loan, divided by the borrower’s self-reported monthly income.|
|delinq_2yrs|The number of 30+ days past-due incidences of delinquency in the borrower's credit file for the past 2 years|
|fico_range_low|The lower boundary range the borrower’s FICO at loan origination belongs to.|
|fico_range_high|The upper boundary range the borrower’s FICO at loan origination belongs to.|
|inq_last_6mths|The number of inquiries in past 6 months (excluding auto and mortgage inquiries)|
|open_acc|The number of open credit lines in the borrower's credit file.|
|pub_rec|Number of derogatory public records|
|revol_bal|Total credit revolving balance|
|revol_util|Revolving line utilization rate, or the amount of credit the borrower is using relative to all available revolving credit.|
|total_acc|The total number of credit lines currently in the borrower's credit file|
|initial_list_status|The initial listing status of the loan. Possible values are – Whole =1, Fractional =0|
|last_fico_range_high|The upper boundary range the borrower’s last FICO pulled belongs to.|
|last_fico_range_low|The lower boundary range the borrower’s last FICO pulled belongs to.|
|collections_12_mths_ex_med|Number of collections in 12 months excluding medical collections|
|policy_code|publicly available policy_code=1 new products not publicly available policy_code=2|
|acc_now_delinq|The number of accounts on which the borrower is now delinquent.|
|tot_coll_amt|Total collection amounts ever owed|
|tot_cur_bal|Total current balance of all accounts|
|total_rev_hi_lim|Total revolving high credit/credit limit|
|acc_open_past_24mths|Number of trades opened in past 24 months.|
|avg_cur_bal|Average current balance of all accounts|
|bc_open_to_buy|Total open to buy on revolving bankcards.|
|bc_util|Ratio of total current balance to high credit/credit limit for all bankcard accounts.|
|chargeoff_within_12_mths|Number of charge-offs within 12 months|
|delinq_amnt|The past-due amount owed for the accounts on which the borrower is now delinquent.|
|mo_sin_old_rev_tl_op|Months since oldest revolving account opened|
|mo_sin_rcnt_rev_tl_op|Months since most recent revolving account opened|
|mo_sin_rcnt_tl|Months since most recent account opened|
|mort_acc|Number of mortgage accounts.|
|mths_since_recent_bc|Months since most recent bankcard account opened.|
|num_accts_ever_120_pd|Number of accounts ever 120 or more days past due|
|num_actv_bc_tl|Number of currently active bankcard accounts|
|num_actv_rev_tl|Number of currently active revolving trades|
|num_bc_sats|Number of satisfactory bankcard accounts|
|num_bc_tl|Number of bankcard accounts|
|num_il_tl|Number of installment accounts|
|num_op_rev_tl|Number of open revolving accounts|
|num_rev_accts|Number of revolving accounts|
|num_rev_tl_bal_gt_0|Number of revolving trades with balance >0|
|num_sats|Number of satisfactory accounts|
|num_tl_30dpd|Number of accounts currently 30 days past due (updated in past 2 months)|
|num_tl_90g_dpd_24m|Number of accounts 90 or more days past due in last 24 months|
|num_tl_op_past_12m|Number of accounts opened in past 12 months|
|pct_tl_nvr_dlq|Percent of trades never delinquent|
|percent_bc_gt_75|Percentage of all bankcard accounts > 75% of limit.|
|pub_rec_bankruptcies|Number of public record bankruptcies|
|tax_liens|Number of tax liens|
|tot_hi_cred_lim|Total high credit/credit limit|
|total_bal_ex_mort|Total credit balance excluding mortgage|
|total_bc_limit|Total bankcard high credit/credit limit|
|total_il_high_credit_limit|Total installment high credit/credit limit|
|joint_app|Indicates whether the loan is an individual application or a joint application with two co-borrowers where joint application = 1 and individual application = 0|
|home_ownership_**|Home ownership status of applicant is mortgage, rent, own, other, none|
|verification_status_Source Verified|Indicates if income was source was verified|
|verification_status_Verified|Indicates if income was verified|
|purpose_**|Purpose of loan is one of the following: credit card, debt, consolidation, educational, improvement, house, major purchase, medical, moving, renewable energy, small business, vacation, wedding, other|
|addr_state_**|Applicant's state of residence|
|loan_status|Is loan good or bad, where good = 1 and bad = 0|
|model_preds|Binary predictions made by model where good = 1 and bad = 0|
|pred_probability|Prediction Probabilities from model where the closer the decimal to 1 the more probable that it is a good loan|

### Findings:

This project was meant to streamline loan application processing so the main objective was to come up with a reliable minimize the number of loans that had to be reviewed by personnel of the bank by running a machine learning model. The baseline score was .872 determined by the good loans as a percentage of all loans. After running a GradientBoostingClassifier model from Scikit-learn I found that the accuracy score of the model was .929 and the Precision score was .956. This is a significant improvement over the .872 basand calling the pred_probability() method, I found that 73% of the loans had a predicted probability of .95 or higher.

This lead me to further analysis one the remaining 27% of the loans. I created another dataframe comprised of all loans with a predicted probability between .25 and .95 and I found that of that portion there was a significant drop off in the precision score.

I also found that there was a reasonable specification score to justify looking at a portion of the dataset with loans with a prediction probability of .25 or less. So the using the predictions for loans with prediction probabilities NOT in the range .25 to .95 is one way to automate loan processing of a significant portion of loan applications, including both the best prediction probability scores and the worst eliminating the most likely good and bad data from human processing.

Finally, I analyzed the the .95+ prediction probability loans further and found that the Precision Score was .99. This lead me to create the "bad_bias" model that labels 'loan_status' either good or bad based on a .95 probability score.

### Metrics and Feature Importance

**Base Model Metrics**

    Total Loans:  432460
    Total Predicted Good:  380003
    Total Predicted Bad:  52457
    Accuracy Score:  0.929
    Recall Score:  0.963
    Specificity Score:  0.696
    Precision Score:  0.956
    F1 Score:  0.959

**Middle Prediction Probability Loans**

    Total Loans:  90617
    Total Predicted Good:  63179
    Total Predicted Bad:  27438
    Accuracy Score:  0.741
    Recall Score:  0.832
    Specificity Score:  0.563
    Precision Score:  0.786
    F1 Score:  0.809

**Solid Prediction Probability Loans**

    Total Loans:  341843
    Total Predicted Good:  316824
    Total Predicted Bad:  25019
    Accuracy Score:  0.979
    Recall Score:  0.987
    Specificity Score:  0.865
    Precision Score:  0.99
    F1 Score:  0.989

**Maximize Precision Version of Model**

    Total Loans:  432460
    Total Predicted Good:  316824
    Total Predicted Bad:  115636
    Accuracy Score:  0.845
    Recall Score:  0.831
    Specificity Score:  0.941
    Precision Score:  0.99
    F1 Score:  0.904

**Feature Importance**

|Feature|Importance|
|--------|-------|
|loan_amnt|0.021|
|term|0.009|
|sub_grade|0.002|
|annual_inc|0.001|
|loan_status|0.002|
|pymnt_plan|0.002|
|delinq_2yrs|0.0|
|fico_range_low|0.0|
|revol_bal|0.0|
|revol_util|0.001|
|total_acc|0.0|
|initial_list_status|0.631|
|last_fico_range_high|0.326|
|tot_coll_amt|0.0|
|tot_cur_bal|0.0|
|total_rev_hi_lim|0.0|
|avg_cur_bal|0.0|
|delinq_amnt|0.001|
|mo_sin_rcnt_rev_tl_op|0.0|
|num_accts_ever_120_pd|0.0|
|num_op_rev_tl|0.001|
|num_rev_accts|0.0|
|num_tl_op_past_12m|0.001|
|percent_bc_gt_75|0.0|
|joint_app|0.0|
|verification_status_Source Verified|0.0|



### Final conclusion/recommendations:

Considering our goals to automate a good portion of the data while minimizing approvals of bad loans, the best idea seems to use the .95 prediction probability weighted model. Using this model will allow for 73% of all the loans being automatically being passed using this model, however, of that 73% the precision score is .99 which is a very good score for our goals. We could have chosen to rely on the model to automatically mark sub-.25 probability loans as bad, but that would not actually eliminate a particularly significant portion of the total loans. So just using the model for the upper end(.95) seems advisable. One could easily set these thresholds at a variety of different levels according to the specific goals or the project or firm.

### References:
1. Lenders Club 2007-2018 (by Nathan George) - https://www.kaggle.com/wordsforthewise/lending-club?select=accepted_2007_to_2018Q4.csv.gz
2. Data Dictionary - https://www.kaggle.com/wordsforthewise/lending-club/discussion/170691
3. Cleaning Tips - https://www.dataquest.io/blog/machine-learning-preparing-data/
4. What is a 'trade' - https://www.thepennyhoarder.com/investing/lending-club-note-trading/
5. Subgrade order - https://www.lendingclub.com/foliofn/rateDetail.action
6. Datetime Conversion 1 - https://stackoverflow.com/questions/2265357/parse-date-string-and-change-format
7. Datetime Conversion 2 - https://stackoverflow.com/questions/9504356/convert-string-into-date-type-on-python
8. initial_list_status column definitions - https://www.lendacademy.com/lending-club-whole-loan-program-one-year-later/
9. GradientBoostingClassifier - https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html
10. Plotting ROC curve - https://www.youtube.com/watch?v=uVJXPPrWRJ0
11. sns.distplot - https://git.generalassemb.ly/DSIR-412/lesson-classification-metrics-ii
12. Confusion Matrix - https://stackoverflow.com/questions/19233771/sklearn-plot-confusion-matrix-with-labels
13. Confusion Matrix - https://stackoverflow.com/questions/3529666/matplotlib-matshow-labels
14. Confusion Matrix - https://stackoverflow.com/questions/17022154/changing-matshow-xticklabel-position-from-top-to-bottom-of-the-figure
15. Dataframe Copy - https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.copy.html
16. List Copy - https://www.w3schools.com/python/python_ref_list.asp
17. Confusion Matrix - https://stackoverflow.com/questions/21712047/matplotlib-imshow-matshow-display-values-on-plot
18. Confusion Matrix - https://stackoverflow.com/questions/42840044/color-a-specific-bar-in-histogram-using-python