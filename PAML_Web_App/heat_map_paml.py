import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import  os

script_dir = os.path.dirname(os.path.abspath(__file__))
dynamic_csv_path = os.path.join(script_dir, "paml_study_data_2.csv")
dynamic_saved_model_path = os.path.join(script_dir, "paml_trained_model.pth")

# Load logged predictions CSV
df = pd.read_csv(dynamic_csv_path)

# Ensure numeric types for correlation
df['confidence_score'] = df['confidence_score'].astype(float)
df['sentiment_score'] = df['sentiment_score'].astype(float)
df['topic_difficulty'] = df['topic_difficulty'].astype(float)
df['time_spent_minutes'] = df['time_spent_minutes'].astype(float)
df['is_struggling'] = df['is_struggling'].astype(float)
df['days_left'] = df['days_left'].astype(float)
df['study_hours'] = df['study_hours'].astype(float)

df['last_exam_result'] = df['last_exam_result'].map({'failed': 0, 'passed': 1})

# Group by exam result and calculate mean study hours
avg_study_hours = df.groupby('last_exam_result')['study_hours'].mean()

# Optional: rename for clean axis
df.rename(columns={
    'confidence_score': 'Confidence',
    'sentiment_score': 'Sentiment',
    'topic_difficulty': 'Topic Diff',
    'time_spent_minutes': 'Time Spent',
    'is_struggling' : 'Is struggling',
    'study_hours': 'Study Hours'
}, inplace=True)

# Compute correlation matrix
correlation_matrix = df[[
    'Confidence', 'Sentiment', 'Topic Diff', 'Time Spent', 'Is struggling', 'days_left', 'Study Hours'
]].corr()

# Plot heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title("Feature Correlation with predicted Study Hours (PAML model)")
plt.tight_layout()
plt.savefig(r"figures/paml_correlation_heatmap.png", dpi=300)
plt.show()


avg_study_hours.plot(kind='bar', color=['red', 'green'])
plt.xticks([0, 1], ['Failed', 'Passed'], rotation=0)
plt.ylabel("Average Predicted Study Hours")
plt.title("Predicted Study Hours by Exam Result")
plt.tight_layout()
plt.savefig(r"figures/last_exam_result_study_hours.png", dpi=300)
plt.show()