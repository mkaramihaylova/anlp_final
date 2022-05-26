import pandas as pd
from simpletransformers.ner import NERModel, NERArgs
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


# preprocessing

sents = pd.read_csv("connl_test_fixed_whole.csv", encoding="latin1")
sents = sents.fillna(method="ffill")
sents["Sentence #"] = LabelEncoder().fit_transform(sents["Sentence #"])
sents.rename(
    columns={"Sentence #": "sentence_id", "Word": "words", "Tag": "labels"},
    inplace=True,
)
sents["labels"] = sents["labels"].str.upper()
X = sents[["sentence_id", "words"]]
Y = sents["labels"]
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)
train = pd.DataFrame(
    {
        "sentence_id": X_train["sentence_id"],
        "words": X_train["words"],
        "labels": y_train,
    }
)
test = pd.DataFrame(
    {"sentence_id": X_test["sentence_id"], "words": X_test["words"], "labels": y_test}
)

# model training

label = sents["labels"].unique().tolist()

args = NERArgs()
args.num_train_epochs = 2
args.learning_rate = 1e-4
args.overwrite_output_dir = True
args.train_batch_size = 16
args.eval_batch_size = 16

model = NERModel("bert", "bert-base-cased", labels=label, args=args, use_cuda=False)
model.train_model(train, eval_data=test, acc=accuracy_score)
result, model_outputs, preds_list = model.eval_model(test)
print(result)
