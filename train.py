
import os
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report

# Dataset folder
DATASET_DIR = "dataset/GenreClassification_Dataset"

TRAIN_PATH = os.path.join(DATASET_DIR, "train_data.txt")
TEST_PATH = os.path.join(DATASET_DIR, "test_data.txt")
SOLUTION_PATH = os.path.join(DATASET_DIR, "test_data_solution.txt")

os.makedirs("models", exist_ok=True)

# Read training data
print("Loading training dataset...")

train_df = pd.read_csv(
    TRAIN_PATH,
    sep=r"\s*:::\s*",
    engine="python",
    header=None,
    names=["id", "title", "genre", "description"],
    dtype=str
)

# Read testing data
print("Loading testing dataset...")

test_df = pd.read_csv(
    TEST_PATH,
    sep=r"\s*:::\s*",
    engine="python",
    header=None,
    names=["id", "title", "description"],
    dtype=str
)

# Clean missing values
train_df = train_df.dropna(subset=["genre", "description"])
test_df = test_df.dropna(subset=["description"])

train_df["genre"] = train_df["genre"].str.strip().str.lower()

# Prepare input and output
X = train_df["description"].astype(str)
y = train_df["genre"]

print("\nTraining records:", len(train_df))
print("Testing records:", len(test_df))
print("Number of genres:", y.nunique())

# Split training data for validation
X_train, X_val, y_train, y_val = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# Create TF-IDF and LinearSVC pipeline
model = Pipeline([
    (
        "tfidf",
        TfidfVectorizer(
            max_features=50000,
            ngram_range=(1, 2),
            stop_words="english",
            sublinear_tf=True
        )
    ),
    (
        "classifier",
        LinearSVC(
            class_weight="balanced",
            random_state=42
        )
    )
])

# Train validation model
print("\nTraining model...")

model.fit(X_train, y_train)

# Validate model
print("\nEvaluating validation performance...")

val_predictions = model.predict(X_val)

validation_accuracy = accuracy_score(
    y_val,
    val_predictions
)

print(
    "\nValidation Accuracy:",
    round(validation_accuracy * 100, 2),
    "%"
)

print("\nValidation Classification Report:")
print(
    classification_report(
        y_val,
        val_predictions,
        zero_division=0
    )
)

# Train final model using complete training dataset
print("\nTraining final model on complete training data...")

model.fit(X, y)

# Predict genres for test dataset
print("\nPredicting test movie genres...")

test_predictions = model.predict(
    test_df["description"].astype(str)
)

# Save predictions
prediction_df = pd.DataFrame({
    "id": test_df["id"],
    "title": test_df["title"],
    "predicted_genre": test_predictions
})

prediction_df.to_csv(
    "models/movie_predictions.csv",
    index=False
)

# Load official test solutions
print("\nLoading official test solutions...")

solution_df = pd.read_csv(
    SOLUTION_PATH,
    sep=r"\s*:::\s*",
    engine="python",
    header=None,
    names=["id", "title", "genre", "description"],
    dtype=str
)

solution_df["genre"] = solution_df["genre"].str.strip().str.lower()

# Compare predictions with actual genres
comparison = prediction_df.merge(
    solution_df[["id", "genre"]],
    on="id",
    how="inner"
)

if len(comparison) > 0:
    test_accuracy = accuracy_score(
        comparison["genre"],
        comparison["predicted_genre"]
    )

    print(
        "\nOfficial Test Accuracy:",
        round(test_accuracy * 100, 2),
        "%"
    )

    print("\nTest Classification Report:")
    print(
        classification_report(
            comparison["genre"],
            comparison["predicted_genre"],
            zero_division=0
        )
    )
else:
    print("No matching movie IDs found.")

# Save trained model
joblib.dump(
    model,
    "models/movie_genre_model.pkl"
)

print("\nModel training completed successfully!")
print("Model saved: models/movie_genre_model.pkl")
print("Predictions saved: models/movie_predictions.csv")
