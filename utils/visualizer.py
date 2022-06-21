# #################################################################################################################### #
#       visualizer.py                                                                                                  #
#           KElbow and Silhouette visualizers.                                                                         #
# #################################################################################################################### #

from colorama import Style, Fore
from sklearn.metrics import silhouette_score
from yellowbrick.cluster import KElbowVisualizer, SilhouetteVisualizer, InterclusterDistance
from yellowbrick.features import PCA


def k_elbow(data, model, k: tuple, verbose: bool = False, **kwargs):
    visualizer = KElbowVisualizer(model(**kwargs), k=k)
    visualizer.fit(data)
    visualizer.show()
    optimal_k = visualizer.elbow_value_

    if verbose:
        print(f"Optimal number of k cluster: {Fore.LIGHTGREEN_EX}{optimal_k}{Fore.RESET}")

    return optimal_k


def silhouette(data, model, k: tuple, verbose: bool = False, **kwargs):
    scores = []

    # Try each num_k
    for num_k in range(k[0], k[1]):
        # Random state is used for reproducibility
        curr_model = model(**{**kwargs, "n_clusters": num_k})
        model_labels = curr_model.fit_predict(data)

        silhouette_avg = silhouette_score(data, model_labels)
        scores.append({"score": silhouette_avg, "num_k": num_k})

    # Get best model
    scores = sorted(scores, key=lambda s: s["score"])
    best_score = scores[0]["score"]
    optimal_k = scores[0]["num_k"]

    # Visualize
    visualizer = SilhouetteVisualizer(model(n_clusters=optimal_k, **kwargs), colors="yellowbrick")
    visualizer.fit(data)
    visualizer.show()

    if verbose:
        print((
            f"Optimal number of k cluster: {Fore.LIGHTGREEN_EX}{optimal_k}{Fore.RESET} "
            f"{Style.DIM}{Fore.WHITE}(score: {round(best_score, 2)}){Style.RESET_ALL}"
        ))

    return optimal_k


def inter_cluster_distance(data, model):
    visualizer = InterclusterDistance(model)
    visualizer.fit(data)
    visualizer.show()


def pca(x, y, classes):
    visualizer = PCA(scale=True, classes=classes)
    visualizer.fit_transform(x, y)
    visualizer.show()