"""
After training PAML:
1. Learning curve plot  (loss vs epoch)  =   paml_learning_curve.png
2. Feature importance   (linear weights) =   paml_feature_importance.png
"""
import torch, matplotlib.pyplot as plt
import numpy as np
from PAML_model import Recommender  
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
dynamic_csv_path = os.path.join(script_dir, "paml_study_data_2.csv")
dynamic_saved_model_path = os.path.join(script_dir, "paml_trained_model.pth")


feature_names = [
    "confidence_score", "sentiment_score", "topic_difficulty",
    "time_spent", "days_left", "exam_known_flag", "exam_result_flag"
]

# load packed model
ckpt = torch.load(dynamic_saved_model_path, map_location="cpu")
loss_history  = ckpt["loss_history"]
state_dict    = ckpt["model_state"]

# learning curve
plt.figure()
plt.plot(range(1, len(loss_history) + 1), loss_history, lw=2)
plt.xlabel("Epoch")
plt.ylabel("Meta Loss")
plt.title("PAML Meta Loss vs Epoch")
plt.grid(True)
plt.tight_layout()
plt.savefig(r"figures/paml_learning_curve.png", dpi=300)
plt.show()
print("Saved learning curve AS paml_learning_curve.png")

# feature study
model = Recommender()
model.load_state_dict(state_dict)
weights = model.fc.weight.detach().numpy().flatten()  # shape (7,)

plt.figure()
plt.barh(feature_names, weights)
plt.xlabel("Weight (positive = more study hours)")
plt.title("PAML Feature Importance (Linear Weights)")
plt.tight_layout()
plt.savefig(r"figures/paml_feature_importance.png", dpi=300)
plt.show()
print("Saved feature importance AS paml_feature_importance.png")
