# import mlflow
# import mlflow.sklearn
# from sklearn.datasets import load_iris
# from sklearn.model_selection import train_test_split, GridSearchCV
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.svm import SVC
# from sklearn.linear_model import LogisticRegression
# from sklearn.metrics import accuracy_score
# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
# import time

# # Thiết lập MLflow
# mlflow.set_experiment("multi_algo_gridsearch")

# # Chuẩn bị dữ liệu
# iris = load_iris()
# X, y = iris.data, iris.target
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # Định nghĩa các thuật toán và lưới tham số
# models = {
#     "RandomForest": {
#         "model": RandomForestClassifier(random_state=42),
#         "params": {
#             "n_estimators": [50, 100, 200],
#             "max_depth": [None, 5, 10]
#         }
#     },
#     "SVM": {
#         "model": SVC(random_state=42),
#         "params": {
#             "C": [0.1, 1, 10],
#             "kernel": ["linear", "rbf"]
#         }
#     },
#     "LogisticRegression": {
#         "model": LogisticRegression(random_state=42, max_iter=1000),
#         "params": {
#             "C": [0.1, 1, 10],
#             "solver": ["lbfgs", "liblinear"]
#         }
#     }
# }

# # Lưu kết quả
# results = []

# # Chạy GridSearchCV cho từng thuật toán
# for model_name, config in models.items():
#     print(f"Training {model_name}...")
    
#     grid = GridSearchCV(
#         estimator=config["model"],
#         param_grid=config["params"],
#         cv=5,
#         scoring="accuracy",
#         n_jobs=-1
#     )
    
#     with mlflow.start_run(run_name=model_name):
#         start_time = time.time()
#         grid.fit(X_train, y_train)
#         training_time = time.time() - start_time
        
#         y_pred = grid.predict(X_test)
#         accuracy = accuracy_score(y_test, y_pred)
        
#         mlflow.log_param("model_name", model_name)
#         for param, value in grid.best_params_.items():
#             mlflow.log_param(param, value)
        
#         mlflow.log_metric("accuracy", accuracy)
#         mlflow.log_metric("training_time", training_time)
        
#         mlflow.sklearn.log_model(grid.best_estimator_, "model")
        
#         results.append({
#             "model_name": model_name,
#             "accuracy": accuracy,
#             "training_time": training_time,
#             "best_params": grid.best_params_
#         })

# # Lưu kết quả thành CSV
# results_df = pd.DataFrame(results)
# results_df.to_csv("model_comparison.csv", index=False)
# mlflow.log_artifact("model_comparison.csv")

# # Vẽ biểu đồ phân tán
# plt.figure(figsize=(8, 6))
# sns.scatterplot(data=results_df, x="training_time", y="accuracy", hue="model_name", size="model_name")
# plt.title("Accuracy vs Training Time")
# plt.xlabel("Training Time (seconds)")
# plt.ylabel("Accuracy")
# plt.savefig("accuracy_vs_time.png")
# mlflow.log_artifact("accuracy_vs_time.png")
# plt.close()

# print("Training completed. Check MLflow UI for details.")



from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC, SVR
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# Load dữ liệu mẫu
X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Khởi tạo mô hình
model = SVC()

# Định nghĩa lưới tham số cần tìm
param_grid = {
    'C': [0.1, 1, 10],
    'kernel': ['linear', 'rbf']
}

# Tạo GridSearchCV
grid_search = GridSearchCV(model, param_grid, cv=5)
grid_search.estimator = SVR()
print(grid_search.estimator)