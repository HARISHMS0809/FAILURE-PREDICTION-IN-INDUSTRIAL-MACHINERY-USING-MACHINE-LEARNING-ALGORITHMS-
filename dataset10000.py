import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler, PowerTransformer
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline as sklearnPipeline

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline

from sklearn.metrics import f1_score, recall_score, precision_score, accuracy_score, make_scorer, confusion_matrix
from sklearn.metrics import roc_curve as sklearn_roc_curve, auc, RocCurveDisplay
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_predict, cross_val_score, train_test_split, RandomizedSearchCV
from sklearn.feature_selection import SelectKBest, mutual_info_classif, f_classif

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.dummy import DummyClassifier

def check_nulls_dupes(df):
  print(f"The amount of nulls: {df.isna().sum()}")
  print(f"The amount of Dupes: {df.duplicated().sum()}")

def check_class_imbalance(target):
  unique, counts = np.unique(target, return_counts=True)
  
  plt.figure(figsize=(10, 7))
  plt.pie(counts, labels=unique, startangle=140, autopct="%1.1f%%")
  plt.title("Target Class Breakdown")
  plt.legend()
  plt.show()
  
  plt.figure(figsize=(10, 7))
  sns.countplot(x=target)
  plt.xlabel("Target Class Breakdown")
  plt.ylabel("Counts")
  plt.show()

def plot_distribution(df, kind):
  
  plt.figure(figsize=(16,16))
  rows = len(df.columns)
  dims = (rows+ 3)//4
  for idx, col in enumerate(df.columns):
    plt.subplot(dims, 4, idx+1)
    sns.histplot(df[col], kde=True) if kind == "Hist" else plt.boxplot(df[col])
    plt.title(f"Distribuition of {col}")
    plt.ylabel("Counts")
  plt.tight_layout()
  plt.show()

def aggregate_dataset(df, interested_columns, agg_col, function):
  
  plt.figure(figsize=(16, 16))
  rows = len(interested_columns)
  dims = (rows + 3 )//4
  
  for idx, col in enumerate(interested_columns):
    grouped_df = getattr(df.groupby(agg_col)[col], function)().reset_index(name=col)
    plt.subplot(dims, 4, idx+1)
    sns.barplot(data=grouped_df, x=agg_col, y=col)
    plt.title(f"Agg of {col}")
    plt.ylabel(col)
    plt.xticks(rotation =45)
  
  plt.tight_layout()
  plt.show()

def plot_confusion_matrix(y_true, y_pred, model_name):
  
  plt.figure(figsize=(10, 7))
  cm = confusion_matrix(y_true, y_pred)
  sns.heatmap(data=cm, annot=True, fmt="d", cmap="Blues")
  plt.title(f"Confusion Matrix For: {model_name}")
  plt.ylabel("Predicted Labels")
  plt.xlabel("True Labels")
  plt.show()

def plot_roc_curve(X,y, sample_model, model, kbest):
  
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42,stratify=y)
  pipeline = create_pipeline(sample_model, model, kbest)
  
  pipeline.fit(X_train, y_train)
  y_scores = pipeline.predict_proba(X_test)[:, 1]
  
  fpr, tpr, thresholds = sklearn_roc_curve(y_test, y_scores)
  roc_auc = auc(fpr, tpr)
  print(f"The AUC is: {roc_auc:.2f}")

  plt.figure()
  lw = 2
  plt.plot(fpr, tpr, color='darkorange', lw=lw, label=f'ROC curve (area = {roc_auc:.2f})')
  plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
  plt.xlim([0.0, 1.0])
  plt.ylim([0.0, 1.05])
  plt.xlabel('False Positive Rate')
  plt.ylabel('True Positive Rate')
  plt.title('Receiver Operating Characteristic')
  plt.legend(loc="lower right")
  plt.show()

def create_pipeline(sample_model, model, kbest=None):
  
  steps = [
    ("Scaler", MinMaxScaler()),
    ("PowerTransformer", PowerTransformer()),
    ("sample", sample_model),
    ("model", model)
  ]
  
  if kbest:
    steps.insert(2, ("Feature-Selector", kbest))
  
  return Pipeline(steps=steps)

def create_sklearn_pipeline( model):
  
  steps = [
    ("Scaler", MinMaxScaler()),
    ("PowerTransformer", PowerTransformer()),
    ("model", model)
  ]
  
  return sklearnPipeline(steps=steps)

def grab_selected_models(names):
  
  models = {
    "SVC": SVC(),
    "LR":LogisticRegression(max_iter=1000),
    "KNN": KNeighborsClassifier(),
    "DTC": DecisionTreeClassifier(),
    "GBC":GradientBoostingClassifier(),
    "RFC":RandomForestClassifier(),
    "XGB": XGBClassifier(),
    "DUMMY": DummyClassifier(strategy="stratified")
  }
  
  return [models[model_name] for model_name in names]

def get_metrics(y, predictions):
    acc_metric = accuracy_score(y, predictions)
    recall_metric = recall_score(y, predictions, average='weighted')
    precision_metric = precision_score(y, predictions, average='weighted')
    f1_metric = f1_score(y, predictions, average='weighted')
    
    return [acc_metric,recall_metric, precision_metric, f1_metric]

def evaluate_model(model, X, y, metric):
  
  cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=10,random_state=1)
  scores = cross_val_score(model, X, y, cv=cv, scoring=metric, n_jobs=-1)
  preds = cross_val_predict(model, X, y, n_jobs=-1, cv=10)
  acc, recall_metric, precision_metric, f1_metric = get_metrics(y, preds)
  
  return [
    round(np.mean(scores),3), 
    round(np.var(scores), 3), 
    round(np.std(scores),3), 
    round(acc,3), 
    round(recall_metric,3),
    round(precision_metric,3),
    round(f1_metric,3)
  ]

def test_selected_models(sample_model, model_names, models, X, y, scoring_metric, kbest=None):
  
  metric_tracker = []
  
  for model, model_name in zip(models, model_names):
    pipeline = create_pipeline(sample_model, model, kbest) if kbest else create_pipeline(sample_model, model) 
    scores = evaluate_model(pipeline, X, y, scoring_metric)
    
    metric_tracker.append({
      "Model": model_name,
      "Mean": scores[0],
      "Var": scores[1],
      "STD": scores[2],
      "Test-Acc":scores[3],
      "recall-Score":scores[4],
      "precision-Score":scores[5],
      "F1-Score":scores[6]
    })
  
  performance_df = pd.DataFrame(metric_tracker).sort_values(by="Mean", ascending=False)
  print(performance_df)

def test_kbest_columns(X, y, sample_model, model, model_name, kbest):
  
  metric_tracker = []
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, stratify=y, random_state=42)
  for k in range(1, X.shape[1]+1):
    metric = SelectKBest(score_func=kbest, k=k)
    pipeline = create_pipeline(sample_model, model, metric)
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    acc, recall_metric, precision_metric, f1_metric = get_metrics(y_test, y_pred)

    metric_tracker.append({
        "Model": model_name,
        "K":k,
        "acc_metric": acc,
        "f1_metric": f1_metric,
        "recall":recall_metric,
        "precision":precision_metric
      })
  
  return pd.DataFrame(metric_tracker)

def pca_analysis(X):
  
  features = range(1, X.shape[1]+1)
  metric_tracker = []
  
  for n in features:
    pca = PCA(n_components=n)
    pipeline = create_sklearn_pipeline(pca)
    X_pca = pipeline.fit_transform(X)
    
    cumsum = np.sum(pca.explained_variance_ratio_)
    
    metric_tracker.append({
      "Component":n,
      "CumSum":cumsum
    }
    )
    
  performance_df = pd.DataFrame(metric_tracker)
  plt.figure(figsize=(12, 8))
  plt.title("PCA Analysis")
  sns.barplot(data=performance_df, x="Component", y="CumSum")
  plt.xlabel("Components")
  plt.ylabel("CumSum")
  plt.show()
  print(performance_df)

def optimization_search(sample_model, model_names, models,X, y, optimizer_class, param_distributions, scoring_metric, kbest):
  
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, stratify=y, random_state=42)
  metric_tracker = []
  
  for model_name, model in zip(model_names, models):
    model_pipeline = create_pipeline(sample_model,model, kbest)
    current_params = param_distributions.get(model_name, {})
    opt_search= optimizer_class(model_pipeline, param_distributions=current_params, cv=10, scoring=scoring_metric, n_jobs=-1)
    opt_search.fit(X_train, y_train)
    
    best_model = opt_search.best_estimator_
    best_params = opt_search.best_params_
    y_pred = best_model.predict(X_test)
    
    acc, recall_metric, precision_metric, f1_metric = get_metrics(y_test, y_pred)
      
    plot_confusion_matrix(y_test, y_pred, model_name)
    print(f"The Best Param: {best_params}")
    metric_tracker.append({
        "Model": model_name,
        "Test-Acc":acc,
        "F1-Score":f1_metric,
        "recall-Score":recall_metric,
        "precision-Score":precision_metric,
      })
  
  return pd.DataFrame(metric_tracker)

def test_voting_classifier(X,y, model, sample_model, kbest, model_name):
  
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, stratify=y, random_state=42)
  metric_tracker = []
  model_pipeline = create_pipeline(sample_model,model, kbest)
  model_pipeline.fit(X_train, y_train)
  y_pred = model_pipeline.predict(X_test)
    
  acc, recall_metric, precision_metric, f1_metric = get_metrics(y_test, y_pred)
      
  plot_confusion_matrix(y_test, y_pred, model_name)
  metric_tracker.append({
        "Model": model_name,
        "Test-Acc":acc,
        "F1-Score":f1_metric,
        "recall-Score":recall_metric,
        "precision-Score":precision_metric,
      })
  
  return pd.DataFrame(metric_tracker)


df = pd.read_csv("ai4i2020.csv")
check_nulls_dupes(df)
df.drop_duplicates(inplace=True)
#check_class_imbalance(df["fail"])
#df.describe().T
#df.dtypes
relations = df.corr()
#plt.figure(figsize=(12, 7))
#sns.heatmap(data=relations, annot=True)
#plot_distribution(df,"Hist")
#interested_columns = df.drop("fail", axis=1).columns

#aggregate_dataset(df, interested_columns, "fail", "sum")
#interested_columns = df.drop("fail", axis=1).columns

#aggregate_dataset(df, interested_columns, "fail", "mean")
X = df.drop("Machine failure", axis=1)
import matplotlib.pyplot as plt
import pandas as pd
x=df.drop(["Machine failure"],axis=1)
y=df["Machine failure"]

count_class = y.value_counts() # Count the occurrences of each class
plt.bar(count_class.index, count_class.values)
plt.xlabel('Class')
plt.ylabel('Count')
plt.title('Class Distribution')
plt.xticks(count_class.index, ['Class 0', 'Class 1'])
plt.show()
y = df["Machine failure"]
from imblearn.over_sampling import SMOTE

smote=SMOTE(sampling_strategy='minority') 
X,y=smote.fit_resample(X,y)
print(y.value_counts())
count_class = y.value_counts() # Count the occurrences of each class
plt.bar(count_class.index, count_class.values)
plt.xlabel('Class')
plt.ylabel('Count')
plt.title('Class Distribution')
plt.xticks(count_class.index, ['Class 0', 'Class 1'])
plt.show()
"""acc_metric = make_scorer(accuracy_score, greater_is_better=True)
f1_metric = make_scorer(f1_score, greater_is_better=True, average="weighted")
precision_metric = make_scorer(precision_score, greater_is_better=True, average='weighted')
recall_metric = make_scorer(recall_score, greater_is_better=True, average='weighted')
model_name = ["DUMMY"]
models = grab_selected_models(model_name)
sample_model = SMOTE()
test_selected_models(sample_model, model_name, models, X, y, acc_metric)
kbest = f_classif
model_name = ["LR"]
models = grab_selected_models(model_name)
test_kbest_columns(X, y, sample_model, models[0], model_name[0], kbest)
kbest = mutual_info_classif
test_kbest_columns(X, y, sample_model, models[0], model_name[0], kbest)
kbest = SelectKBest(score_func=mutual_info_classif, k=4)
model_names = ["LR", "SVC", "KNN"]
models = grab_selected_models(model_names)
test_selected_models(sample_model, model_names, models, X, y, f1_metric, kbest)
model_names = ["RFC", "XGB", "GBC"]
models = grab_selected_models(model_names)
test_selected_models(sample_model, model_names, models, X, y, f1_metric, kbest)
params = {
    'GBC': {
        'model__n_estimators': [100, 200, 300],
        'model__learning_rate': [0.01, 0.1, 0.2],
        'model__max_depth': [3, 4, 5],
        'model__min_samples_split': [2, 4],
        'model__min_samples_leaf': [1, 2]
    },
    'XGB': {
        'model__n_estimators': [100, 150, 200],
        'model__learning_rate': [0.01, 0.05, 0.1],
        'model__max_depth': [3, 5, 7],
        'model__subsample': [0.6, 0.8, 1.0],
        'model__colsample_bytree': [0.6, 0.8, 1.0]
    },
    'RFC': {
        'model__criterion':["gini", "entropy"],
        'model__n_estimators': [100, 200, 300],
        'model__max_depth': [None, 10, 20],
        'model__min_samples_split': [2, 5, 10],
        'model__min_samples_leaf': [1, 2, 4]
    },
    'LR': {
        'model__C': [0.01, 0.1, 1, 10],
        'model__penalty': ['l1', 'l2'],
        'model__solver': ['liblinear', 'saga']
    },
    'SVC': {
        'model__C': [0.1, 1, 10],
        'model__kernel': ['linear', 'rbf'],
        'model__gamma': ['scale', 'auto'],
        'model__degree': [2, 3, 4]
    },
    'KNN': {
    'model__n_neighbors': [3, 5, 10, 15],
    'model__weights': ['uniform', 'distance'],
    'model__algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
    'model__leaf_size': [10, 30, 50],
    'model__p': [1, 2]
}
}
model_names = ["LR", "SVC", "KNN"]
models = grab_selected_models(model_names)
performance_df = optimization_search(sample_model, model_names, models, X, y, RandomizedSearchCV, params, f1_metric, kbest)
print(performance_df.sort_values(by="Test-Acc", ascending=False))
model_names = ["RFC", "GBC", "XGB"]
models = grab_selected_models(model_names)
performance_df = optimization_search(sample_model, model_names, models, X, y, RandomizedSearchCV, params, f1_metric, kbest)
print(performance_df.sort_values(by="Test-Acc", ascending=False))
best_models = [
  ("XGB", XGBClassifier(subsample=0.8, n_estimators=100, max_depth=5, learning_rate=0.05, colsample_bytree=0.6)),
  ("GBC", GradientBoostingClassifier(n_estimators=200, min_samples_split=4, min_samples_leaf=2, max_depth=5, learning_rate=0.01)),
  ("RFC", RandomForestClassifier(n_estimators=200, min_samples_split=10, min_samples_leaf=4, max_depth=4, criterion='entropy')),
  ("KNN", KNeighborsClassifier(weights='uniform', p=2, n_neighbors=3, leaf_size=50, algorithm='kd_tree')),
  ("LR", LogisticRegression(solver="saga", penalty='l2', C=0.1)),
  ("SVC", SVC(kernel='rbf', gamma="auto", degree=4, C=0.1, probability=True))
]
clf = VotingClassifier(estimators=best_models,voting="soft")
test_voting_classifier(X,y, clf, sample_model, kbest, "Voting-Soft")

clf = VotingClassifier(estimators=best_models,voting="hard")
test_voting_classifier(X,y, clf, sample_model, kbest, "Voting-Hard")
clf = VotingClassifier(estimators=best_models,voting="soft")
plot_roc_curve(X, y, sample_model, clf, kbest)"""
