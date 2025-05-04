from sklearn.metrics import classification_report
from tabulate import tabulate
import pandas as pd

y_true = [0, 1, 2, 2, 2]
y_pred = [0, 0, 2, 2, 1]

report = classification_report(y_true, y_pred, output_dict=True)
df = pd.DataFrame(report).transpose()

markdown_table = tabulate(df, headers="keys", tablefmt="github", floatfmt=".2f")

with open("my/classification_report.md", "w") as f:
    f.write("## 📊 Classification Report\n\n")
    f.write("Here's a beautifully formatted classification report:\n\n")
    f.write(markdown_table)