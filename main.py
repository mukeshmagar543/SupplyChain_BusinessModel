from sklearn.model_selection import train_test_split
from src.SPC.data_loader import load_data
from src.SPC.preprocessing import preprocess, apply_pca
from src.SPC.model_trainer import get_model
from src.SPC.evaluator import evaluate_model
import joblib


def main():
    df = load_data()
    df = preprocess(df)
    pca_df = apply_pca(df)

    X = pca_df.drop(columns='product_wg_ton')
    y = pca_df['product_wg_ton']

    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.70)

    model = get_model()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_train)

    score = evaluate_model(y_train, y_pred)
    print("Training R^2 Score:", score)


if __name__ == "__main__":
    main()
