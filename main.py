
from sklearn.model_selection import train_test_split
from data.simulate_data import generate_bot_human_data
from models.lightgbm_model import train_lgbm
from utils.metrics import evaluate_model

from sklearn.model_selection import train_test_split

def main():
    print("Starting Twitter Bot Detection...\n")

    df = generate_bot_human_data()
    X = df.drop("is_bot", axis=1)
    y = df["is_bot"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print("Training model...\n")
    model = train_lgbm(X_train, y_train)

    y_pred = model.predict(X_test)

    print("Evaluation results:\n")
    evaluate_model(y_test, y_pred)

if __name__ == "__main__":
    main()
