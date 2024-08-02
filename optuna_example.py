import optuna
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from optuna.samplers import TPESampler

# Load the iris dataset
data = load_iris()
X = data.data
y = data.target

# Objective function to be minimized
def objective(trial):
    # Define the hyperparameters and their ranges
    n_estimators = trial.suggest_int('n_estimators', 10, 200)
    max_depth = trial.suggest_int('max_depth', 1, 20)
    min_samples_split = trial.suggest_int('min_samples_split', 2, 20)
    min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 20)
    bootstrap = trial.suggest_categorical('bootstrap', [True, False])

    # Create a random forest classifier with the current hyperparameters
    rf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        bootstrap=bootstrap,
        random_state=42
    )

    # Evaluate the classifier using cross-validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    score = cross_val_score(rf, X, y, cv=cv, scoring='accuracy').mean()

    return score

# Create a study object and optimize the objective function
study = optuna.create_study(direction='maximize', sampler=TPESampler(seed=42))
study.optimize(objective, n_trials=100, n_jobs=-1)

# Print the best parameters and best score
print(f"Best parameters found: {study.best_params}")
print(f"Best score found: {study.best_value}")
