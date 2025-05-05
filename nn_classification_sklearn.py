from typing import Tuple
import numpy as np
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay
from sklearn.model_selection import GridSearchCV
from matplotlib import pyplot as plt
import warnings
# We will suppress ConvergenceWarnings in this task. In practice, you should take warnings more seriously.
warnings.filterwarnings("ignore")


def reduce_dimension(X_train: np.ndarray, n_components: int) -> Tuple[np.ndarray, PCA]:
    """
    :param X_train: Training data to reduce the dimensionality. Shape: (n_samples, n_features)
    :param n_components: Number of principal components
    :return: Data with reduced dimensionality, which has shape (n_samples, n_components), and the PCA object
    """

    # TODO: Create a PCA object and fit it using X_train - Created below
    #       Transform X_train using the PCA object. = Done
    #       Print the explained variance ratio of the PCA object. - Done
    #       Return both the transformed data and the PCA object. - Done
    # Create PCA object with specified number of components and random_state
    pca = PCA(n_components=n_components, random_state=42)
    # Fit PCA on training data
    pca.fit(X_train)
    # Transform training data
    X_train_pca = pca.transform(X_train)
    # Print explained variance ratio
    explained_variance = sum(pca.explained_variance_ratio_) * 100
    print(f"Explained variance with {n_components} components: {explained_variance:.2f}%")
    
    return X_train_pca, pca


def train_nn(X_train: np.ndarray, y_train: np.ndarray) -> MLPClassifier:
    """
    Train MLPClassifier with different number of neurons and hidden layers.

    :param X_train: PCA-projected features with shape (n_samples, n_components)
    :param y_train: Targets
    :return: The MLPClassifier you consider to be the best
    """

    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train,
                                                      test_size=0.2, random_state=42)

    # TODO: Train MLPClassifier with different number of layers/neurons. - |Completed
    #       Print the train accuracy, validation accuracy, and the training loss for each configuration. Done 
    #       Return the MLPClassifier that you consider to be the best. Done 

    # Define the different hidden layer architectures to try
    hidden_layer_architectures = [
        (2,), (8,), (64,), (256,), (1024,), (128, 256, 128)
    ]
    
    # Store results for analysis
    results = {}
    
    # Try each architecture
    for hidden_layers in hidden_layer_architectures:
        print(f"\nTraining neural network with hidden layers: {hidden_layers}")
        
        # Create MLPClassifier
        clf = MLPClassifier(
            hidden_layer_sizes=hidden_layers,
            max_iter=100,
            solver='adam',
            random_state=1
        )
        
        # Train the model
        clf.fit(X_train, y_train)
        
        # Evaluate on train and validation sets
        train_acc = clf.score(X_train, y_train)
        val_acc = clf.score(X_val, y_val)
        final_loss = clf.loss_
        
        # Store results
        results[hidden_layers] = {
            'classifier': clf,
            'train_accuracy': train_acc,
            'val_accuracy': val_acc,
            'final_loss': final_loss
        }
        
        print(f"Train accuracy: {train_acc:.4f}")
        print(f"Validation accuracy: {val_acc:.4f}")
        print(f"Final loss: {final_loss:.4f}")
    
    # Find best model based on validation accuracy
    best_hidden_layers = max(results.keys(), key=lambda k: results[k]['val_accuracy'])
    best_classifier = results[best_hidden_layers]['classifier']
    
    print(f"\nBest architecture: {best_hidden_layers}")
    print(f"Best validation accuracy: {results[best_hidden_layers]['val_accuracy']:.4f}")
    
    return best_classifier


def train_nn_with_regularization(X_train: np.ndarray, y_train: np.ndarray) -> MLPClassifier:
    """
    Train MLPClassifier using regularization.

    :param X_train: PCA-projected features with shape (n_samples, n_components)
    :param y_train: Targets
    :return: The MLPClassifier you consider to be the best
    """
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train,
                                                      test_size=0.2, random_state=42)

    # TODO: Use the code from the `train_nn` function, but add regularization to the MLPClassifier.
    #       Again, return the MLPClassifier that you consider to be the best.
    # Define the different hidden layer architectures to try
    hidden_layer_architectures = [
        (2,), (8,), (64,), (256,), (1024,), (128, 256, 128)
    ]
    
    # Choose one of the following settings:
    # (a) alpha = 0.1
    # (b) early_stopping = True
    # (c) alpha = 0.1 and early_stopping = True
    # For this implementation, we'll use option (c)
    
    alpha = 0.1
    early_stopping = True
    
    # Store results for analysis
    results = {}
    
    # Try each architecture
    for hidden_layers in hidden_layer_architectures:
        print(f"\nTraining neural network with hidden layers: {hidden_layers}, alpha={alpha}, early_stopping={early_stopping}")
        
        # Create MLPClassifier with regularization
        clf = MLPClassifier(
            hidden_layer_sizes=hidden_layers,
            max_iter=100,
            solver='adam',
            random_state=1,
            alpha=alpha,  # L2 regularization
            early_stopping=early_stopping  # Early stopping
        )
        
        # Train the model
        clf.fit(X_train, y_train)
        
        # Evaluate on train and validation sets
        train_acc = clf.score(X_train, y_train)
        val_acc = clf.score(X_val, y_val)
        final_loss = clf.loss_
        
        # Store results
        results[hidden_layers] = {
            'classifier': clf,
            'train_accuracy': train_acc,
            'val_accuracy': val_acc,
            'final_loss': final_loss
        }
        
        print(f"Train accuracy: {train_acc:.4f}")
        print(f"Validation accuracy: {val_acc:.4f}")
        print(f"Final loss: {final_loss:.4f}")
    
    # Find best model based on validation accuracy
    best_hidden_layers = max(results.keys(), key=lambda k: results[k]['val_accuracy'])
    best_classifier = results[best_hidden_layers]['classifier']
    
    print(f"\nBest architecture with regularization: {best_hidden_layers}")
    print(f"Best validation accuracy: {results[best_hidden_layers]['val_accuracy']:.4f}")
    
    return best_classifier


def plot_training_loss_curve(nn: MLPClassifier) -> None:
    """
    Plot the training loss curve.

    :param nn: The trained MLPClassifier
    """
    # TODO: Plot the training loss curve of the MLPClassifier. Don't forget to label the axes. - Done
    plt.figure(figsize=(10, 6))
    plt.plot(nn.loss_curve_, label='Training Loss')
    plt.title('Training Loss Over Iterations')
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def show_confusion_matrix_and_classification_report(nn: MLPClassifier, X_test: np.ndarray, y_test: np.ndarray) -> None:
    """
    Plot confusion matrix and print classification report.

    :param nn: The trained MLPClassifier you want to evaluate
    :param X_test: Test features (PCA-projected)
    :param y_test: Test targets
    """
    # TODO: Use `nn` to compute predictions on `X_test`. - Done
    #       Use `confusion_matrix` and `ConfusionMatrixDisplay` to plot the confusion matrix on the test data. - Done
    #       Use `classification_report` to print the classification report. - Done 
    # Get predictions
    y_pred = nn.predict(X_test)
    
    # Calculate accuracy
    accuracy = nn.score(X_test, y_test)
    print(f"Test accuracy: {accuracy:.4f}")
    
    # Create confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    
    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=nn.classes_)
    disp.plot(cmap=plt.cm.Blues, values_format='d')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.show()
    
    # Print classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))


def perform_grid_search(X_train: np.ndarray, y_train: np.ndarray) -> MLPClassifier:
    """
    Perform GridSearch using GridSearchCV.

    :param X_train: PCA-projected features with shape (n_samples, n_components)
    :param y_train: Targets
    :return: The best estimator (MLPClassifier) found by GridSearchCV
    """
    # TODO: Create parameter dictionary for GridSearchCV, as specified in the assignment sheet.
    param_grid = {
        'alpha': [0.0, 0.1, 1.0],
        'batch_size': [32, 512],
        'hidden_layer_sizes': [(128,), (256,)]
    }
     # Calculate the number of different architectures
    num_architectures = len(param_grid['alpha']) * len(param_grid['batch_size']) * len(param_grid['hidden_layer_sizes'])
    print(f"Number of different architectures to check: {num_architectures}")
    print(f"Calculation: {len(param_grid['alpha'])} alphas * {len(param_grid['batch_size'])} batch sizes * {len(param_grid['hidden_layer_sizes'])} hidden layer configurations = {num_architectures} combinations")
    #       Create an MLPClassifier with the specified default values.
    clf = MLPClassifier(
        max_iter=100,
        solver='adam',
        random_state=42
    )    

    #       Run the grid search with `cv=5` and (optionally) `verbose=4`.
    grid_search = GridSearchCV(
        estimator=clf,
        param_grid=param_grid,
        cv=5,  # 5-fold cross-validation
        verbose=4
    )
    print("Performing grid search...")
    grid_search.fit(X_train, y_train)

    #       Print the best score (mean cross validation score) and the best parameter set.
    print("\nBest parameters found:")
    print(grid_search.best_params_)
    print(f"Best cross-validation score: {grid_search.best_score_:.4f}")
    #       Return the best estimator found by GridSearchCV.
    return grid_search.best_estimator_