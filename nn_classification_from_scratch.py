from sklearn.model_selection import train_test_split
from mlp_classifier_own import MLPClassifierOwn
import numpy as np

def train_nn_own(X_train: np.ndarray, y_train: np.ndarray) -> MLPClassifierOwn:
    """
    Train MLPClassifierOwn with PCA-projected features.

    :param X_train: PCA-projected features with shape (n_samples, n_components)
    :param y_train: Targets
    :return: The MLPClassifierOwn object
    """
    # Set random seed for reproducibility
    np.random.seed(24)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train,
                                                      test_size=0.2, random_state=42)

    # TODO: Create a MLPClassifierOwn object and fit it using (X_train, y_train)
    #       Print the train accuracy and validation accuracy
    #       Return the trained model
    classifier = MLPClassifierOwn(
        num_epochs=5,
        alpha=0.0,
        batch_size=32,
        hidden_layer_sizes=(16,),
        random_state=42
    )
    # Fit the model on the training data
    print("Training neural network from scratch...")
    classifier.fit(X_train, y_train)
    
    # Evaluate the model
    train_accuracy = classifier.score(X_train, y_train)
    val_accuracy = classifier.score(X_val, y_val)
    
    print(f"Train accuracy: {train_accuracy:.4f}")
    print(f"Validation accuracy: {val_accuracy:.4f}")
    
    return classifier