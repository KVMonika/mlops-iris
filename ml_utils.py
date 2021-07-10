from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

# define a Gaussain NB classifier
clf = GaussianNB()
# other classifiers
clf2 = DecisionTreeClassifier()
clf3 = KNeighborsClassifier(n_neighbors=7, metric='euclidean')
clf4 = make_pipeline(StandardScaler(),
                    SGDClassifier(max_iter=1000, tol=1e-3))

# define the class encodings and reverse encodings
classes = {0: "Iris Setosa", 1: "Iris Versicolour", 2: "Iris Virginica"}
r_classes = {y: x for x, y in classes.items()}

# function to train and load the model during startup
def load_model():
    # load the dataset from the official sklearn datasets
    X, y = datasets.load_iris(return_X_y=True)

    # do the test-train split and train the model
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    clf.fit(X_train, y_train)
    clf2.fit(X_train, y_train)
    clf3.fit(X_train, y_train)
    clf4.fit(X_train, y_train)

    # calculate the print the accuracy score
    acc = accuracy_score(y_test, clf.predict(X_test))
    acc2 = accuracy_score(y_test, clf2.predict(X_test))
    acc3 = accuracy_score(y_test, clf3.predict(X_test))
    acc4 = accuracy_score(y_test, clf4.predict(X_test))

    print(f"Model trained with accuracy: {round(acc, 3)}")
    print(f"Model2 trained with accuracy: {round(acc2, 3)}")
    print(f"Model3 trained with accuracy: {round(acc3, 3)}")
    print(f"Model4 trained with accuracy: {round(acc4, 3)}")

# function to predict the flower using the model
def predict(query_data):
    x = list(query_data.dict().values())
    # Observation: Out of all 4 classifiers, KNN performed well on all relaods compared to inconsistent accuaracy from other models
    # So using KNN to predict
    prediction = clf3.predict([x])[0]
    print(f"Model prediction: {classes[prediction]}")
    return classes[prediction]

# function to retrain the model as part of the feedback loop
def retrain(data):
    # pull out the relevant X and y from the FeedbackIn object
    X = [list(d.dict().values())[:-1] for d in data]
    y = [r_classes[d.flower_class] for d in data]

    # fit the classifier again based on the new data obtained
    clf3.fit(X, y)
