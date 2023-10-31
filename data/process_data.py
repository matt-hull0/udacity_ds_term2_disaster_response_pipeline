import sys
import pandas as pd
from sqlalchemy import create_engine


def load_data(messages_filepath: str, categories_filepath: str) -> pd.DataFrame:
    """load messages and associated categories data from csv files

    Parameters
    ----------
    messages_filepath : str

    categories_filepath : str

    Returns
    -------
    pd.DataFrame

    """
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = messages.set_index("id").join(categories.set_index("id"), how="inner")

    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean category column ready for MLP

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    pd.DataFrame
    with categories split into single column for each category
    """

    categories = df.categories.str.split(";", expand=True)

    row = categories.iloc[0]
    category_colnames = [value[:-2] for value in row]
    categories.columns = category_colnames

    for column in categories:
        categories[column] = categories[column].str[-1]
        categories[column] = categories[column].astype(int)

    categories = categories.replace(2, 1)

    df = df.drop(columns=["categories"])
    df = df.join(categories)
    df = df.drop_duplicates()

    return df


def save_data(df: pd.DataFrame, database_filename: str) -> None:
    """save dataframe in sqlite database

    Parameters
    ----------
    df : pd.DataFrame_
    database_filename : str
    """

    engine = create_engine("sqlite:///" + database_filename)
    df.to_sql("DisasterMessagesTable", engine, index=False, if_exists="replace")


def main():
    """Process Data Script"""
    if len(sys.argv) == 4:
        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print(
            "Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}".format(
                messages_filepath, categories_filepath
            )
        )
        df = load_data(messages_filepath, categories_filepath)

        print("Cleaning data...")
        df = clean_data(df)

        print("Saving data...\n    DATABASE: {}".format(database_filepath))
        save_data(df, database_filepath)

        print("Cleaned data saved to database!")

    else:
        print(
            "Please provide the filepaths of the messages and categories "
            "datasets as the first and second argument respectively, as "
            "well as the filepath of the database to save the cleaned data "
            "to as the third argument. \n\nExample: python process_data.py "
            "disaster_messages.csv disaster_categories.csv "
            "DisasterResponse.db"
        )


if __name__ == "__main__":
    main()
