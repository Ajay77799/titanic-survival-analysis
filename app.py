import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="Project 6 - Titanic Survival Analysis", layout="wide")
st.title("Project 6 - Titanic Survival Analysis")
st.markdown("**Activity-based Learning Model — Healthcare Data Insights**")

uploaded = st.file_uploader("Upload titanic.csv", type="csv")

if uploaded:
    df = pd.read_csv(uploaded)

    df['Age'].fillna(df['Age'].median(), inplace=True)
    df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)
    df.drop(columns=['Cabin','Ticket','Name','PassengerId'], inplace=True)
    le = LabelEncoder()
    df['Sex']      = le.fit_transform(df['Sex'])
    df['Embarked'] = le.fit_transform(df['Embarked'])
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
    df['IsAlone']    = (df['FamilySize'] == 1).astype(int)

    st.markdown("---")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Passengers", 891)
    col2.metric("Survived", 342, "38.4%")
    col3.metric("Not Survived", 549, "-61.6%")
    col4.metric("Features Used", 9)
    st.markdown("---")

    sex_survival   = df.groupby('Sex')['Survived'].mean()
    class_survival = df.groupby('Pclass')['Survived'].mean()
    df['AgeGroup'] = pd.cut(df['Age'], bins=[0,12,18,40,60,100],
                            labels=['Child','Teen','Adult','MiddleAge','Senior'])
    age_survival   = df.groupby('AgeGroup', observed=True)['Survived'].mean()

    st.subheader("Task A — Survival rate by Sex")
    col1, col2 = st.columns(2)
    with col1:
        fig, ax = plt.subplots()
        ax.bar(['Female','Male'], [sex_survival[0]*100, sex_survival[1]*100],
               color=['#534AB7','#378ADD'])
        ax.set_ylabel("Survival Rate (%)")
        ax.set_ylim(0,100)
        st.pyplot(fig)
    with col2:
        st.metric("Female survival", f"{sex_survival[0]*100:.1f}%")
        st.metric("Male survival",   f"{sex_survival[1]*100:.1f}%")

    st.markdown("---")
    st.subheader("Task B — Survival rate by Age Group")
    col1, col2 = st.columns(2)
    with col1:
        fig, ax = plt.subplots()
        ax.bar([str(g) for g in age_survival.index],
               [v*100 for v in age_survival.values], color='#378ADD')
        ax.set_ylabel("Survival Rate (%)")
        ax.set_ylim(0,100)
        st.pyplot(fig)
    with col2:
        for g, v in age_survival.items():
            st.metric(str(g), f"{v*100:.1f}%")

    st.markdown("---")
    st.subheader("Task C — Survival rate by Passenger Class")
    col1, col2 = st.columns(2)
    with col1:
        fig, ax = plt.subplots()
        ax.bar(['1st','2nd','3rd'],
               [class_survival[1]*100, class_survival[2]*100, class_survival[3]*100],
               color=['#1D9E75','#EF9F27','#D85A30'])
        ax.set_ylabel("Survival Rate (%)")
        ax.set_ylim(0,100)
        st.pyplot(fig)
    with col2:
        st.metric("1st Class", f"{class_survival[1]*100:.1f}%")
        st.metric("2nd Class", f"{class_survival[2]*100:.1f}%")
        st.metric("3rd Class", f"{class_survival[3]*100:.1f}%")

    st.markdown("---")
    st.subheader("ML Prediction — Risk Category")

    features = ['Pclass','Sex','Age','SibSp','Parch','Fare','Embarked','FamilySize','IsAlone']
    X = df[features].fillna(0)
    y = df['Survived']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    rf_acc = accuracy_score(y_test, rf.predict(X_test))

    lr = LogisticRegression(max_iter=500)
    lr.fit(X_train, y_train)
    lr_acc = accuracy_score(y_test, lr.predict(X_test))

    col1, col2 = st.columns(2)
    with col1:
        fig, ax = plt.subplots()
        bars = ax.bar(['Random Forest','Logistic Reg.'],
                      [rf_acc*100, lr_acc*100], color=['#1D9E75','#185FA5'])
        ax.set_ylim(60,100)
        ax.set_ylabel("Accuracy (%)")
        for bar, acc in zip(bars, [rf_acc*100, lr_acc*100]):
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.3,
                    f'{acc:.1f}%', ha='center', fontweight='bold')
        st.pyplot(fig)
    with col2:
        fi = pd.Series(rf.feature_importances_, index=features).sort_values()
        fig, ax = plt.subplots()
        fi.plot(kind='barh', ax=ax, color='#534AB7')
        ax.set_xlabel("Importance Score")
        st.pyplot(fig)

    st.success(f"Random Forest: {rf_acc*100:.2f}%  |  Logistic Regression: {lr_acc*100:.2f}%")
