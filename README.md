# Импорт необходимых библиотек
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
# Загрузка датасета "Br east Cancer"
data = load_breast_cancer()
X = data.data
y = data.target
# Разделение данных на обучающий и тестовый набор
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Создание базовой модели (слабой модели) - дерево решений
base_model = DecisionTreeClassifier(max_depth=1)
# Создание модели AdaBoost
model = AdaBoostClassifier(base_model, n_estimators=50, random_state=42)
# Обучение модели
model.fit(X_train, y_train)
# Предсказание на тестовых данных
y_pred = model.predict(X_test)
# Оценка точности
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy}')
# Вывод отчета по классификации
print(classification_report(y_test, y_pred))
# Вывод матрицы ошибок
confusion = confusion_matrix(y_test, y_pred)
print('Confusion Matrix:')
print(confusion)
