from timeit import default_timer as timer
from datetime import timedelta
import colorama
from colorama import Style, Fore
import pandas
from sklearn.cluster import KMeans
import utils


def main():
    program_start = timer()

    # Read the CSV
    data = pandas.read_csv("./data/anthems.csv", delimiter=",", dtype={
        "Country": str,
        "Alpha-2": str,
        "Alpha-3": str,
        "Continent": str,
        "Anthem": str
    })

    # First look
    utils.print.title("First look")
    utils.dataframe.first_look(data)

    # Missing values
    utils.print.title("Missing values")
    utils.dataframe.missing_values(data, keep_zeros=False)

    # Keep only interesting columns
    utils.print.title("Keep only interesting columns")
    models_data = data[["Anthem"]]
    print(models_data.sample(n=5))

    # Text preprocessing
    utils.print.title("Text preprocessing")

    def text_preprocessing(text):
        text_cleaned = utils.text.clean(text)
        token = utils.text.tokenize(text_cleaned, language="english")
        token = utils.text.remove_stopwords(token, language="english")
        token = utils.text.remove_punctuation(token)

        # PorterStemmer seems to produce less stable model, but it has a much better
        # inter-cluster distance than WordNetLemmatizer.
        token = utils.text.stem(token)

        return token

    vectorized_data = utils.text.vectorization(models_data, col="Anthem", analyzer=text_preprocessing)
    tfidf_vectorized_data = utils.text.tfidf_vectorization(models_data, col="Anthem", analyzer=text_preprocessing)

    print(f"Vectorized:\n{Style.DIM}{Fore.WHITE}{vectorized_data.head()}")
    print(f"TFIDF Vectorized:\n{Style.DIM}{Fore.WHITE}{tfidf_vectorized_data.head()}")

    """data_x = utils.text.tfidf_vectorization(models_data, col="Anthem", analyzer=text_preprocessing)
    data_y = models_data["Continent"]
    x_train, x_test, y_train, y_test = train_test_split(data_x, data_y, test_size=0.2)
    print(x_train)
    print(y_train)"""

    # KMeans
    utils.print.title("Text preprocessing")
    model_kwargs = {"init": "k-means++"}

    utils.print.title("KElbowVisualizer", char="~")
    elbow_optimal_k = utils.visualizer.k_elbow(tfidf_vectorized_data, KMeans, k=(2, 20), verbose=True, **model_kwargs)

    utils.print.title("SilhouetteVisualizer", char="~")
    silhouette_optimal_k = utils.visualizer.silhouette(tfidf_vectorized_data, KMeans, k=(2, 20), verbose=True, **model_kwargs)

    utils.print.title("Optimal model", char="~")
    optimal_k = int(round(((elbow_optimal_k + silhouette_optimal_k) / 2), 0))
    model_kwargs["n_clusters"] = optimal_k

    print(f"Using {Fore.LIGHTGREEN_EX}n_clusters={optimal_k}{Fore.RESET} for KMeans()")
    k_means = KMeans(**model_kwargs)
    k_fit = k_means.fit(tfidf_vectorized_data)

    data["Cluster"] = k_fit.labels_
    print(data.sample(n=10))

    utils.visualizer.inter_cluster_distance(tfidf_vectorized_data, KMeans(**model_kwargs))
    # utils.visualizer.pca(vectorized_data, data[["Cluster"]], classes=data["Cluster"].drop_duplicates().tolist())

    clusters_map = utils.plot.Map()
    utils.plot.MapLayer("Cluster per countries", show_default=True)\
        .add_to(clusters_map)\
        .load_dataframe(data)\
        .to_choropleth(
            geo_data=f"{utils.plot.folium_data}/world-countries.json",
            columns=["Alpha-3", "Cluster"],
            name="Cluster per countries",
            legend_name="LEGEND",
        )

    clusters_map.open(notebook=False)

    utils.plot.generate_wordcloud(data["Anthem"], title="Le Wordcloud")

    # Program end
    program_end = timer()
    program_elapsed_time = timedelta(seconds=program_end - program_start)
    print(f"\n{Fore.LIGHTGREEN_EX}Successful processing of \"anthems.csv\" in {program_elapsed_time}.")


if __name__ == "__main__":
    colorama.init(autoreset=True)
    main()
