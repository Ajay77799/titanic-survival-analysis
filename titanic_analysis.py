import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import warnings
warnings.filterwarnings('ignore')

# ── YOUR FILE PATH ──────────────────────────────────────────
CSV_PATH = "/Users/ajaysiphone/Library/Mobile Documents/com~apple~CloudDocs/Downloads/titanic.csv"
# ────────────────────────────────────────────────────────────

print("=" * 60)
print("  PROJECT 6 - TITANIC SURVIVAL ANALYSIS")
print("=" * 60)

# STEP 1: Load
print("\n[STEP 1] Loading dataset...")
df = pd.read_csv(CSV_PATH)
print(f"  Rows: {df.shape[0]}  |  Columns: {df.shape[1]}")

# STEP 2: Explore
print("\n[STEP 2] Exploring data...")
print(f"  Columns : {list(df.columns)}")
print(f"  Missing values:\n{df.isnull().sum()[df.isnull().sum()>0].to_string()}")
print(f"\n  Survival count:\n{df['Survived'].value_counts().to_string()}")

# STEP 3: Clean
print("\n[STEP 3] Cleaning data...")
df['Age'].fillna(df['Age'].median(), inplace=True)
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)
df.drop(columns=['Cabin','Ticket','Name','PassengerId'], inplace=True)
le = LabelEncoder()
df['Sex']      = le.fit_transform(df['Sex'])       # female=0, male=1
df['Embarked'] = le.fit_transform(df['Embarked'])  # C=0, Q=1, S=2
print(f"  Done. Shape after cleaning: {df.shape}")

# STEP 4: Feature Engineering
print("\n[STEP 4] Feature engineering...")
df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
df['IsAlone']    = (df['FamilySize'] == 1).astype(int)
df['AgeGroup']   = pd.cut(df['Age'], bins=[0,12,18,40,60,100],
                           labels=['Child','Teen','Adult','MiddleAge','Senior'])
print("  Created: FamilySize, IsAlone, AgeGroup")

# STEP 5: Analysis
print("\n[STEP 5] Analysis (Project 6 Tasks)...")

sex_survival   = df.groupby('Sex')['Survived'].mean().round(3)
class_survival = df.groupby('Pclass')['Survived'].mean().round(3)
age_survival   = df.groupby('AgeGroup', observed=True)['Survived'].mean().round(3)

print(f"\n  Task A - Survival rate by Sex:")
print(f"    Female : {sex_survival[0]*100:.1f}%")
print(f"    Male   : {sex_survival[1]*100:.1f}%")

print(f"\n  Task B - Survival rate by Age group:")
for g, v in age_survival.items():
    print(f"    {str(g):<12}: {v*100:.1f}%")

print(f"\n  Task C - Survival rate by Passenger Class:")
for c, v in class_survival.items():
    print(f"    Class {c} : {v*100:.1f}%")

# STEP 6: ML Prediction
print("\n[STEP 6] Training ML models...")
features = ['Pclass','Sex','Age','SibSp','Parch','Fare','Embarked','FamilySize','IsAlone']
X = df[features].fillna(0)
y = df['Survived']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"  Train: {len(X_train)} samples | Test: {len(X_test)} samples")

rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
rf_acc = accuracy_score(y_test, rf.predict(X_test))

lr = LogisticRegression(max_iter=500)
lr.fit(X_train, y_train)
lr_acc = accuracy_score(y_test, lr.predict(X_test))

print(f"\n  Random Forest Accuracy     : {rf_acc*100:.2f}%")
print(f"  Logistic Regression Accuracy: {lr_acc*100:.2f}%")

print("\n  Classification Report (Random Forest):")
print(classification_report(y_test, rf.predict(X_test),
      target_names=['Not Survived','Survived']))

fi = pd.Series(rf.feature_importances_, index=features).sort_values(ascending=False)
print("  Feature Importance:")
for feat, score in fi.items():
    bar = '█' * int(score * 80)
    print(f"    {feat:<14}: {bar} {score*100:.1f}%")

# STEP 7: Visualizations
print("\n[STEP 7] Saving charts to ~/titanic_project/titanic_results.png ...")

fig, axes = plt.subplots(2, 3, figsize=(15, 9))
fig.suptitle('Project 6 - Titanic Survival Analysis', fontsize=14, fontweight='bold')

axes[0,0].bar(['Not Survived','Survived'],
              df['Survived'].value_counts().sort_index(),
              color=['#D85A30','#1D9E75'], edgecolor='white')
axes[0,0].set_title('Overall survival')

axes[0,1].bar(['Female','Male'],
              [sex_survival[0]*100, sex_survival[1]*100],
              color=['#534AB7','#378ADD'], edgecolor='white')
axes[0,1].set_title('Survival rate by sex')
axes[0,1].set_ylabel('Survival rate (%)')
axes[0,1].set_ylim(0,100)

axes[0,2].bar(['1st','2nd','3rd'],
              [class_survival[1]*100, class_survival[2]*100, class_survival[3]*100],
              color=['#1D9E75','#EF9F27','#D85A30'], edgecolor='white')
axes[0,2].set_title('Survival rate by class')
axes[0,2].set_ylabel('Survival rate (%)')
axes[0,2].set_ylim(0,100)

axes[1,0].bar([str(g) for g in age_survival.index],
              [v*100 for v in age_survival.values],
              color='#378ADD', edgecolor='white')
axes[1,0].set_title('Survival rate by age group')
axes[1,0].set_ylabel('Survival rate (%)')
axes[1,0].set_ylim(0,100)

fi.plot(kind='barh', ax=axes[1,1], color='#534AB7', edgecolor='white')
axes[1,1].set_title('Feature importance')
axes[1,1].set_xlabel('Score')

bars = axes[1,2].bar(['Random Forest','Logistic Reg.'],
                     [rf_acc*100, lr_acc*100],
                     color=['#1D9E75','#185FA5'], edgecolor='white')
axes[1,2].set_title('Model accuracy comparison')
axes[1,2].set_ylabel('Accuracy (%)')
axes[1,2].set_ylim(60,100)
for bar, acc in zip(bars, [rf_acc*100, lr_acc*100]):
    axes[1,2].text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.3,
                   f'{acc:.1f}%', ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
output_path = '/Users/ajaysiphone/titanic_project/titanic_results.png'
plt.savefig(output_path, dpi=150, bbox_inches='tight')
plt.close()

print("=" * 60)
print("  DONE!")
print(f"  Random Forest : {rf_acc*100:.2f}%")
print(f"  Logistic Reg. : {lr_acc*100:.2f}%")
print(f"  Chart saved   : {output_path}")
print("=" * 60)

