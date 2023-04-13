from happytransformer import HappyWordPrediction

model = HappyWordPrediction("BERT", load_path="HooshvareLab/bert-base-parsbert-uncased")
model.save("/mnt/disk1/users/naziri/unbiased_model")
