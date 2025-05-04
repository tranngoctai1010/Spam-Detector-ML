from sklearn.metrics import classification_report
from tabulate import tabulate
import pandas as pd

y_true = [0, 1, 2, 2, 2]
y_pred = [0, 0, 2, 2, 1]

report_dict = classification_report(y_true, y_pred, output_dict=True)
report_df = pd.DataFrame(report_dict).transpose()
print(tabulate(report_df, headers='keys', tablefmt='grid', floatfmt=".2f"))
