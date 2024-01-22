import numpy as np
import pandas as pd
import torch


# SEQUENCE GENERATION
PADDING_VALUE = float('-inf')


def generate_input_sequence(logo_embeddings: pd.DataFrame, null_features: int, sequence_length: int,
                            is_randomized: bool, is_padding: bool) -> torch.Tensor:
    """
    Build a torch tensor for the transformer input sequences.
    Includes
    - Randomization (optional)
    - Generation of padding

    Args:
        logo_embeddings (pd.DataFrame): DataFrame containing logo embeddings.
        null_features (int): Number of null features to add to each embedding.
        sequence_length (int): Target length for padding sequences.
        is_randomized: shuffle order of paths
        is_padding: if true, function adds padding

    Returns:
        torch.Tensor: Tensor representing the input sequences.
    """
    logo_embeddings.drop(columns=['filename', 'animation_id'], inplace=True)

    # Randomization
    if is_randomized:
        logo_embeddings = logo_embeddings.sample(frac=1).reset_index(drop=True)

    # Null Features
    if null_features > 0:
        logo_embeddings = pd.concat([logo_embeddings,
                                     pd.DataFrame(0,
                                                  index=logo_embeddings.index,
                                                  columns=range(logo_embeddings.shape[1],
                                                                logo_embeddings.shape[1] + null_features))],
                                    axis=1,
                                    ignore_index=True)

    if is_padding:
        logo_embeddings = _add_padding(logo_embeddings, sequence_length)

    return torch.tensor(logo_embeddings.values)


def generate_output_sequence(animation: pd.DataFrame, sequence_length: int, is_randomized: bool,
                             is_padding: bool) -> torch.Tensor:
    """
    Build a torch tensor for the transformer output sequences.
    Includes
    - Randomization (later, when same start time)
    - Generation of padding
    - Add EOS Token

    Args:
        animation (pd.DataFrame): DataFrame containing logo embeddings.
        sequence_length (int): Target length for padding sequences.
        is_randomized: shuffle order of paths, applies when same start time
        is_padding: if true, function adds padding

    Returns:
        torch.Tensor: Tensor representing the input sequences.
    """
    if is_randomized:
        animation = animation.sample(frac=1).reset_index(drop=True)
        print("Note: Randomization not implemented yet")

    animation.sort_values(by=['a13'], inplace=True)  # again ordered by time start.
    animation.drop(columns=['file', 'filename'], inplace=True)

    # Append the EOS row to the DataFrame
    sos_eos_row = {col: 0 for col in animation.columns}
    sos_eos_row["a0"] = 1
    sos_eos_row = pd.DataFrame([sos_eos_row])
    animation = pd.concat([sos_eos_row, animation, sos_eos_row],
                          ignore_index=True)

    # Padding Generation: Add padding rows or cut off excess rows
    if is_padding:
        animation = _add_padding(animation, sequence_length)

    return torch.Tensor(animation.values)


def _add_padding(dataframe: pd.DataFrame, sequence_length: int) -> pd.DataFrame:
    if len(dataframe) < sequence_length:
        padding_rows = pd.DataFrame([[PADDING_VALUE] * len(dataframe.columns)] * (sequence_length - len(dataframe)),
                                    columns=dataframe.columns)
        dataframe = pd.concat([dataframe, padding_rows], ignore_index=True)
    elif len(dataframe) > sequence_length:
        # Cut off excess rows
        dataframe = dataframe.iloc[:sequence_length]

    return dataframe


# BUCKETING
def generate_buckets_2D(dataset, column1, column2, quantiles1, quantiles2, print_histogram=True):
    """

    Args:
        dataset: dataset to generate buckets for
        column1: first column name
        column2: second column name
        quantiles1: initial quantiles for column1
        quantiles2: initial quantiles for column2
        print_histogram: if true, a histogram of the 2D buckets is printed

    Returns: dictionary object with bucket edges

    """
    x_edges = dataset[column1].quantile(quantiles1)
    y_edges = dataset[column2].quantile(quantiles2)

    x_edges = np.array(x_edges)
    y_edges = np.unique(y_edges)

    if print_histogram:
        hist, x_edges, y_edges = np.histogram2d(dataset[column1],
                                                dataset[column2],
                                                bins=[x_edges, y_edges])
        print(hist)

    return {
        "input_edges": list(x_edges),
        "output_edges": list(y_edges)
    }


def get_bucket(input_length, output_length, buckets):
    bucket_name = ""

    for i, input_edge in enumerate(buckets["input_edges"]):
        # print(f"{i}: {input_length} < {input_edge}")
        if input_length > input_edge:
            continue

        bucket_name = bucket_name + str(int(i))  # chr(ord('A')+i)
        break

    bucket_name = bucket_name + "-"

    for i, output_edge in enumerate(buckets["output_edges"]):
        if output_length > output_edge:
            continue

        bucket_name = bucket_name + str(int(i))
        break

    return bucket_name
