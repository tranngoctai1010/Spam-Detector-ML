import mlflow
import mlflow.sklearn
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris  # Thay bằng dữ liệu email

def train_and_track(X, y, model, param_grid, scoring='accuracy', cv=5, experiment_name="Spam_Detector"):
    """Huấn luyện mô hình và lưu vào MLflow."""
    mlflow.set_experiment(experiment_name)
    
    with mlflow.start_run(run_name=f"{model.__class__.__name__}_Run"):
        # Huấn luyện với GridSearchCV
        grid_search = GridSearchCV(model, param_grid, cv=cv, scoring=scoring)
        grid_search.fit(X, y)
        
        # Lưu kết quả
        best_model = grid_search.best_estimator_
        best_score = grid_search.best_score_
        best_params = grid_search.best_params_
        cv_results = pd.DataFrame(grid_search.cv_results_)
        
        # Lưu vào MLflow
        mlflow.log_param("model_name", model.__class__.__name__)
        mlflow.log_param("param_grid", str(param_grid))
        mlflow.log_metric("best_score", best_score)
        for param, value in best_params.items():
            mlflow.log_param(f"best_{param}", value)
        mlflow.sklearn.log_model(best_model, "best_model")
        
        # Lưu cv_results và biểu đồ
        cv_results.to_csv("cv_results.csv")
        mlflow.log_artifact("cv_results.csv")
        
        plt.figure(figsize=(8, 6))
        sns.boxplot(data=cv_results, x=cv_results.columns[-2], y="mean_test_score")
        plt.title(f"CV Results for {model.__class__.__name__}")
        plt.savefig("cv_results_plot.png")
        plt.close()
        mlflow.log_artifact("cv_results_plot.png")
        
        return {
            "best_model": best_model,
            "best_score": best_score,
            "best_params": best_params,
            "run_id": mlflow.active_run().info.run_id
        }

def deploy_model(run_id, port=1234):
    """Triển khai mô hình thành API với MLflow."""
    import os
    os.system(f"mlflow models serve -m runs:/{run_id}/best_model -p {port} --no-conda")

if __name__ == "__main__":
    # Tải dữ liệu (thay bằng dữ liệu email của Spam-Detector-ML)
    X, y = load_iris().data, load_iris().target
    
    # Định nghĩa mô hình và lưới tham số
    model = SVC()
    param_grid = {'C': [0.1, 1, 10], 'kernel': ['linear', 'rbf']}
    
    # Huấn luyện và lưu vào MLflow
    results = train_and_track(X, y, model, param_grid)
    print(f"Best Score: {results['best_score']}, Best Params: {results['best_params']}")
    
    # Triển khai API (chạy trong terminal hoặc background)
    # deploy_model(results['run_id'])  # Uncomment để triển khai













from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np

def regression_report(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100  # MAPE in percentage
    
    report = {
        'MSE': mse,
        'RMSE': rmse,
        'MAE': mae,
        'R²': r2,
        'MAPE (%)': mape
    }
    
    # In báo cáo dạng bảng
    print("Regression Report:")
    print("-" * 30)
    for metric, value in report.items():
        print(f"{metric:<10} : {value:.3f}")
    return report

# Dữ liệu mẫu
y_true = np.array([3.0, -0.5, 2.0, 7.0])
y_pred = np.array([2.5, 0.0, 2.1, 7.8])
report = regression_report(y_true, y_pred)


# Regression Report:
# ------------------------------
# MSE        : 0.375
# RMSE       : 0.612
# MAE        : 0.425
# R²         : 0.948
# MAPE (%)   : 17.361