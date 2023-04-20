from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.model_selection import train_test_split
import graphviz

# load iris dataset
iris = load_iris()

# split dataset into train and test set
X_train, X_test, y_train, y_test = train_test_split(
    iris.data, iris.target, test_size=0.3, random_state=42)

# create decision tree classifier
clf = DecisionTreeClassifier(max_depth=3)

# train classifier
clf.fit(X_train, y_train)

# predict on test set
y_pred = clf.predict(X_test)

# evaluate classifier
accuracy = clf.score(X_test, y_test)
print(f"Accuracy: {accuracy:.2f}")

# generate visualization of decision tree
dot_data = export_graphviz(clf, out_file=None,
                           feature_names=iris.feature_names,
                           class_names=iris.target_names,
                           filled=True, rounded=True,
                           special_characters=True)
graph = graphviz.Source(dot_data)
graph.render("iris_decision_tree")
graph