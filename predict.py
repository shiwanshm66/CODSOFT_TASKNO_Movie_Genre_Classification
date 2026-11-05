
import joblib

# Load trained model
model = joblib.load("models/movie_genre_model.pkl")

print("\n===================================")
print("      MOVIE GENRE PREDICTOR")
print("===================================")

while True:
    plot = input("\nEnter movie plot (or type 'exit'): ")

    if plot.lower() == "exit":
        print("Thank you for using Movie Genre Predictor!")
        break

    if not plot.strip():
        print("Please enter a valid movie plot.")
        continue

    prediction = model.predict([plot])

    print("\nPredicted Movie Genre:", prediction[0])
    print("-----------------------------------")
