from dataclasses import dataclass
from typing import List, Tuple, Any, Optional

@dataclass
class PreprocessingConfig:
    steps: List[Tuple[str, Any]]  # Các bước tiền xử lý (tên, transformer)
    param_grid: Optional[dict[str, List[Any]]] = None  # Tham số tối ưu cho preprocessing

@dataclass
class ModelConfig:
    models: List[object]  # Danh sách mô hình ML
    param_grid: List[Tuple[str, Any]]  # Tham số tối ưu cho mô hình

@dataclass
class ModelSelectionConfig:
    selector: object  # Cơ chế chọn mô hình (ví dụ: GridSearchCV)

@dataclass
class AutoMLConfig:
    preprocess_config: PreprocessingConfig
    model_config: ModelConfig
    model_selection_config: ModelSelectionConfig
    
    
    
    
    
    
    
    
    
    
    
    
from dataclasses import dataclass
from typing import List, Tuple, Any, Optional
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
import pandas as pd
import time
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error
import numpy as np
from tabulate import tabulate

@dataclass
class ResultFormatter:
    """Class chịu trách nhiệm định dạng và hiển thị kết quả đánh giá mô hình."""
    results: List[dict]
    display_format: str = "pandas"  # pandas, tabulate, json
    decimals: int = 4

    def to_dataframe(self) -> pd.DataFrame:
        """Chuyển kết quả thành DataFrame."""
        return pd.DataFrame(self.results)

    def print_results(self):
        """In bảng kết quả theo định dạng được chọn."""
        if not self.results:
            raise ValueError("No results to display")

        df = self.to_dataframe()
        if self.display_format == "pandas":
            pd.set_option("display.max_columns", None)
            pd.set_option("display.width", None)
            pd.set_option("display.float_format", f"{{:.{self.decimals}f}}".format)
            print("\n=== Model Evaluation Results ===")
            print(df.to_string(index=False))
            print("================================")
        elif self.display_format == "tabulate":
            print("\n=== Model Evaluation Results ===")
            print(tabulate(df, headers="keys", tablefmt="fancy_grid", floatfmt=f".{self.decimals}f"))
            print("================================")
        elif self.display_format == "json":
            print(df.to_json(orient="records", indent=2))

    def save_results(self, filename: str):
        """Lưu kết quả vào file."""
        df = self.to_dataframe()
        if filename.endswith(".csv"):
            df.to_csv(filename, index=False)
        elif filename.endswith(".json"):
            df.to_json(filename, orient="records")
        else:
            raise ValueError("Unsupported file format. Use .csv or .json")
        print(f"Results saved to {filename}")

@dataclass
class AutoMLConfig:
    models: List[Any]
    preprocess_steps: List[Tuple[str, Any]]
    param_grids: List[dict[str, List[Any]]]
    cv: int = 5
    scoring: Optional[str] = None
    verbose: bool = False
    metrics: Optional[List[str]] = None  # Danh sách chỉ số bổ sung

    def _calculate_metrics(self, y_true, y_pred, scoring: str) -> dict:
        """Tính các chỉ số bổ sung dựa trên scoring."""
        metrics = {}
        if self.metrics:
            for metric in self.metrics:
                if metric == "accuracy" and scoring in ["accuracy", "f1"]:
                    metrics["Accuracy"] = accuracy_score(y_true, y_pred)
                elif metric == "f1" and scoring in ["accuracy", "f1"]:
                    metrics["F1 Score"] = f1_score(y_true, y_pred, average="weighted")
                elif metric == "rmse" and scoring == "neg_mean_squared_error":
                    metrics["RMSE"] = np.sqrt(mean_squared_error(y_true, y_pred))
        return metrics

    def fit(self, X, y, X_test=None, y_test=None):
        """Huấn luyện mô hình và lưu kết quả."""
        results = []
        for model, param_grid in zip(self.models, self.param_grids):
            start_time = time.time()
            
            # Tạo pipeline
            pipeline = Pipeline(self.preprocess_steps + [("model", model)])
            
            # Kết hợp param_grid
            combined_param_grid = {
                **{f"{step[0]}__{k}": v for step in self.preprocess_steps for k, v in param_grid.items() if step[0] in k},
                **{f"model__{k}": v for k, v in param_grid.items() if not any(step[0] in k for step in self.preprocess_steps)}
            }
            
            # Chạy GridSearchCV
            grid_search = GridSearchCV(
                pipeline, 
                combined_param_grid, 
                cv=self.cv, 
                scoring=self.scoring, 
                verbose=1 if self.verbose else 0
            )
            grid_search.fit(X, y)
            
            # Tính thời gian huấn luyện
            training_time = time.time() - start_time
            
            # Tính chỉ số trên tập test (nếu có) hoặc tập huấn luyện
            X_eval, y_eval = (X_test, y_test) if X_test is not None and y_test is not None else (X, y)
            y_pred = grid_search.predict(X_eval)
            
            # Tính chỉ số bổ sung
            additional_metrics = self._calculate_metrics(y_eval, y_pred, self.scoring)
            
            # Lưu kết quả
            results.append({
                "Model": model.__class__.__name__,
                "Best Score": grid_search.best_score_,
                "Best Parameters": grid_search.best_params_,
                "Training Time (s)": training_time,
                **additional_metrics
            })
        
        # Tạo formatter và lưu kết quả
        self.formatter = ResultFormatter(results=results, display_format="tabulate")
        self.formatter.print_results()
        
        return self

    def save_results(self, filename: str):
        """Lưu kết quả vào file."""
        if not hasattr(self, "formatter"):
            raise ValueError("Must call fit() first")
        self.formatter.save_results(filename)

    def get_best_model(self):
        """Trả về mô hình tốt nhất."""
        if not hasattr(self, "formatter"):
            raise ValueError("Must call fit() first")
        df = self.formatter.to_dataframe()
        return df.loc[df["Best Score"].idxmax()]["best_estimator"]

# Ví dụ sử dụng
if __name__ == "__main__":
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.datasets import make_classification

    # Tạo dữ liệu mẫu
    X, y = make_classification(n_samples=100, n_features=20, random_state=42)

    # Định nghĩa AutoMLConfig
    auto_ml = AutoMLConfig(
        models=[RandomForestClassifier(), LogisticRegression()],
        preprocess_steps=[("scaler", StandardScaler())],
        param_grids=[
            {"model__n_estimators": [100, 200], "model__max_depth": [None, 10]},
            {"model__C": [0.1, 1.0]}
        ],
        cv=3,
        scoring="accuracy",
        verbose=True,
        metrics=["accuracy", "f1"]
    )

    # Chạy AutoML và in bảng kết quả
    auto_ml.fit(X, y)

    # Lưu kết quả vào file
    auto_ml.save_results("model_results.csv")
    
    
    
    
    
    
    
    
    
    
    
    
    
    import mlflow
import mlflow.sklearn
from sklearn.model_selection import GridSearchCV
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

class MLflowTracker:
    """Tracker đơn giản cho MLflow."""
    def __init__(self, experiment_name):
        mlflow.set_experiment(experiment_name)

    def track(self, run_name, model_name, param_grid, best_score, best_params, best_model, cv_results):
        with mlflow.start_run(run_name=run_name):
            mlflow.log_param("model_name", model_name)
            mlflow.log_param("param_grid", str(param_grid))
            mlflow.log_metric("best_score", best_score)
            for param, value in best_params.items():
                mlflow.log_param(f"best_{param}", value)
            mlflow.sklearn.log_model(best_model, "best_model")
            cv_results.to_csv("cv_results.csv")
            mlflow.log_artifact("cv_results.csv")
            plt.figure(figsize=(8, 6))
            sns.boxplot(data=cv_results, x=cv_results.columns[-2], y="mean_test_score")  # Param cuối cùng
            plt.title(f"CV Results for {model_name}")
            plt.savefig("cv_results_plot.png")
            plt.close()
            mlflow.log_artifact("cv_results_plot.png")

class AutoTrain:
    """Class tự động huấn luyện mô hình với tracker tùy chọn."""
    def __init__(self, model, param_grid, scoring='accuracy', cv=5, tracker=None):
        self.model = model
        self.param_grid = param_grid
        self.scoring = scoring
        self.cv = cv
        self.tracker = tracker  # Tracker là instance hoặc None
        self.grid_search = None
        self.best_model = None
        self.best_score = None
        self.best_params = None

    def fit(self, X, y):
        """Huấn luyện mô hình và theo dõi nếu có tracker."""
        # Huấn luyện với GridSearchCV
        self.grid_search = GridSearchCV(self.model, self.param_grid, cv=self.cv, scoring=self.scoring)
        self.grid_search.fit(X, y)
        self.best_model = self.grid_search.best_estimator_
        self.best_score = self.grid_search.best_score_
        self.best_params = self.grid_search.best_params_
        cv_results = pd.DataFrame(self.grid_search.cv_results_)

        # Theo dõi nếu có tracker
        if self.tracker:
            self.tracker.track(
                run_name=f"{self.model.__class__.__name__}_Run",
                model_name=self.model.__class__.__name__,
                param_grid=self.param_grid,
                best_score=self.best_score,
                best_params=self.best_params,
                best_model=self.best_model,
                cv_results=cv_results
            )
        else:
            # Lưu cục bộ nếu không có tracker
            cv_results.to_csv("cv_results.csv")
            plt.figure(figsize=(8, 6))
            sns.boxplot(data=cv_results, x=cv_results.columns[-2], y="mean_test_score")
            plt.title(f"CV Results for {self.model.__class__.__name__}")
            plt.savefig("cv_results_plot.png")
            plt.close()

        return self

    def predict(self, X):
        """Dự đoán với mô hình tốt nhất."""
        if self.best_model is None:
            raise ValueError("Model has not been trained yet.")
        return self.best_model.predict(X)

    def get_results(self):
        """Trả về kết quả huấn luyện."""
        return {
            "best_model": self.best_model,
            "best_score": self.best_score,
            "best_params": self.best_params,
            "cv_results": pd.DataFrame(self.grid_search.cv_results_)
        }