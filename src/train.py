import os
import joblib

def main():
    from sklearn.datasets import load_iris
    iris = load_iris()
    X = iris.data      # shape (150, 4)
    y = iris.target    # shape (150,)
    print(iris.feature_names, iris.target_names)

    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    from sklearn.tree import DecisionTreeClassifier
    model = DecisionTreeClassifier(random_state=42)

    model.fit(X_train, y_train)

    os.makedirs("outputs", exist_ok=True)
    joblib.dump(model, "outputs/model.joblib")

    y_pred = model.predict(X_test)

    print("True labels:", y_test[:5])
    print("DecisionTreeClassifier Predictions:", y_pred[:5])


    from sklearn.metrics import accuracy_score
    accuracy = accuracy_score(y_test, y_pred)
    print("DecisionTreeClassifier Accuracy:", accuracy)

    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:\n", cm)

    from sklearn.metrics import ConfusionMatrixDisplay
    cm_display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=iris.target_names)
    cm_display.plot()

    import matplotlib.pyplot as plt
    plt.show()
    plt.imsave("outputs/confmatimg" + ".png", cm_display.figure_.canvas.buffer_rgba())



    from sklearn.neighbors import KNeighborsClassifier
    model2 = KNeighborsClassifier(n_neighbors=5)
    model2.fit(X_train, y_train)
    y_pred2 = model2.predict(X_test)
    print("\n K NeighbourClassifier Predictions:", y_pred2[:5])
    print("k-NN accuracy:", accuracy_score(y_test, y_pred2))

if __name__=="__main__":
    main()

# The end of the python code file
