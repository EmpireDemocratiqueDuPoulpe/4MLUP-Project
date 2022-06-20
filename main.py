from timeit import default_timer as timer
from datetime import timedelta
import colorama
from colorama import Style, Fore
import pandas
import numpy
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
    data = data[["Country", "Anthem"]]
    print(data.sample(n=5))

    # Text preprocessing
    utils.print.title("Text preprocessing")

    def text_preprocessing(text):
        text_cleaned = utils.text.clean(text)
        token = utils.text.tokenize(text_cleaned, language="english")
        token = utils.text.remove_stopwords(token, language="english")
        token = utils.text.remove_punctuation(token)
        token = utils.text.lemmatize(token)

        return token

    vectorized_data = utils.text.vectorization(data, col="Anthem", analyzer=text_preprocessing)
    tfidf_vectorized_data = utils.text.tfidf_vectorization(data, col="Anthem", analyzer=text_preprocessing)

    print(f"Vectorized:\n{Style.DIM}{Fore.WHITE}{vectorized_data.head()}")
    print(f"TFIDF Vectorized:\n{Style.DIM}{Fore.WHITE}{tfidf_vectorized_data.head()}")

    # Program end
    program_end = timer()
    program_elapsed_time = timedelta(seconds=program_end - program_start)
    print(f"\n{Fore.LIGHTGREEN_EX}Successful processing of \"anthems.csv\" in {program_elapsed_time}.")


if __name__ == "__main__":
    colorama.init(autoreset=True)
    main()
