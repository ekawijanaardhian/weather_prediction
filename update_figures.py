import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
feat_df = pd.read_csv('hasil_riset_bmkg/Table_3_Feature_Importance.csv')
top_feat = feat_df.head(8).copy()
top_feat['Importance'] = top_feat['Importance_Formatted'].astype(float)
top_feat['Feature_Clean'] = top_feat['Feature_Clean'].replace({
    'Heat Index (Rothfusz)': 'Heat Index',
    'day_of_year': 'Day of Year'
})
plt.figure(figsize=(9, 5), dpi=300)
sns.barplot(x='Importance', y='Feature_Clean', data=top_feat, hue='Feature_Clean', palette='viridis', legend=False)
plt.title('Feature Importance Analysis (Random Forest Regressor)', fontsize=13, fontweight='bold')
plt.xlabel('Relative Importance Score (Mean Decrease in Impurity)', fontsize=11)
plt.ylabel('Predictor Features', fontsize=11)
plt.tight_layout()
os.makedirs('hasil_riset_bmkg/figures', exist_ok=True)
plt.savefig('hasil_riset_bmkg/figures/fig_feature_importance.png')
plt.close()
