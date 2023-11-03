
# 3.0

def subsets(nums):
    """Generete powerset from a input set

    Args:
        nums (list): set of numbers

    Returns:
        list: The power set of the input set.
    """
    
    rez = set()

    def rec(sub):
        if not sub:
            return
        rez.add(tuple(sub))

        for i in sub:
            new_sub = sub.copy()
            new_sub.remove(i)
            rec(new_sub)

    rec(nums)
    rez = list(rez)
    # rez.append([])
    return rez

# 4.1

def split_data(data, features, target):
    """Split on test/train/val.

    Args:
        df (pd.DataFrame): input dataset
        features (list): list of features
        target (str): target column name
    Returns:
        tuple: X_train, X_test, y_train, y_test, X_val, y_val
    """
    from sklearn.model_selection import train_test_split
    # Split on test/train 80/20
    # split startified on target
    X_train, X_test, y_train, y_test = train_test_split(data[features], data[target], test_size=0.3, random_state=1)
    # make validation set
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=1)
    return X_train, X_test, y_train, y_test, X_val, y_val


def train_model(model, X_train, y_train, X_test, y_test, X_val, y_val):
    """Train model and return predictions.

    Args:
        model (sklearn model): model to train
        X_train (pd.DataFrame): training data
        y_train (pd.DataFrame): training labels
        X_test (pd.DataFrame): test data
        y_test (pd.DataFrame): test labels
        X_val (pd.DataFrame): validation data
        y_val (pd.DataFrame): validation labels
    
    Returns:
        tuple: mae, model
    """
    from sklearn.metrics import mean_absolute_error
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)
    # Evaluaate
    mae = mean_absolute_error(y_val, y_pred)
    return mae, model
