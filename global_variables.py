UNKNOWN_WORD = "_UNK_"
END_WORD = "_END_"
NAN_WORD = "_NAN_"
COMMENT = "comment_text"
LIST_CLASSES = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]

TRAIN_FILENAME = 'assets/raw_data/train.csv'
TEST_FILENAME = 'assets/raw_data/test.csv'
LIST_LOGITS = ['logits_' + c for c in LIST_CLASSES]
