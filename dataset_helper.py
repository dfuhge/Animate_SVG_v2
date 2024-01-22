

import numpy as np


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
