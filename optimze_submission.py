import pandas as pd

PROBABILITIES_NORMALIZE_COEFFICIENT = 1.4
list_classes = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]

orig_submission = pd.read_csv("submissions/pavel11/model_e8v0.0425.csv")
predictions = orig_submission[list_classes]

#predictions **= PROBABILITIES_NORMALIZE_COEFFICIENT
predictions /= 1.1
new_submission = pd.read_csv("assets/raw_data/sample_submission.csv")
new_submission[list_classes] = predictions
new_submission.to_csv("submissions/pavel11/model_e8v0.0425_corrected_div1.1.csv", index=False)
