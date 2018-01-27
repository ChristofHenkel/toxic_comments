import pandas as pd

PROBABILITIES_NORMALIZE_COEFFICIENT = 1.4
list_classes = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
target_means = [0.167311,0.012612,0.102474,0.005164,0.089256,0.017147]

orig_submission = pd.read_csv("submissions/pavel11/model_e8v0.0425.csv")
predictions = orig_submission[list_classes]

#predictions **= PROBABILITIES_NORMALIZE_COEFFICIENT
predictions /= 1.1
new_submission = pd.read_csv("assets/raw_data/sample_submission.csv")
new_submission[list_classes] = predictions
new_submission.to_csv("submissions/pavel11/model_e8v0.0425_corrected_div1.1.csv", index=False)


means = predictions.mean(axis = 0)
means = means.to_dict()
#sds = predictions.std(axis = 0)
#sds = sds.to_dict()

target_means = dict(zip(list_classes,target_means))
#target_std = dict(zip(list_classes,[0.293549,0.051652,0.230786,0.038028,0.193155,0.075126]))

factors = {label:target_means[label]/means[label] for label in means}
new_predictions = predictions.multiply(factors)






"""
Target: 

toxic            0.167311
severe_toxic     0.012612
obscene          0.102474
threat           0.005164
insult           0.089256
identity_hate    0.017147

toxic            0.293549
severe_toxic     0.051652
obscene          0.230786
threat           0.038028
insult           0.193155
identity_hate    0.075126


"""

