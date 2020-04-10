from argparse import ArgumentParser
import pickle

import numpy as np


def print_stats(data, hist=False):
    print("Min/Max/Median/Mean(Std) %f/%f/%f/%f(%f)" % (min(data), max(data), float(np.median(data)),
                                                        float(np.mean(data)), float(np.std(data))))
    if hist:
        hist1 = np.histogram(data)
        for x, y in zip(hist1[0], hist1[1]):
            print("%s %s" % (x, y))


def print_results(results):
    times = np.array(results["times"])
    lens = np.array([len(x) for x in results["solutions"]])
    num_nodes_generated = np.array(results["num_nodes_generated"])

    print("-Times-")
    print_stats(times)
    print("-Lengths-")
    print_stats(lens)
    print("-Nodes Generated-")
    print_stats(num_nodes_generated)
    print("-Nodes/Sec-")
    print_stats(np.array(num_nodes_generated) / np.array(times))


def main():
    # parse arguments
    parser: ArgumentParser = ArgumentParser()
    parser.add_argument('--soln1', type=str, required=True, help="")
    parser.add_argument('--soln2', type=str, required=True, help="")

    args = parser.parse_args()

    results1 = pickle.load(open(args.soln1, "rb"))
    results2 = pickle.load(open(args.soln2, "rb"))

    lens1 = np.array([len(x) for x in results1["solutions"]])
    lens2 = np.array([len(x) for x in results2["solutions"]])

    print("%i states" % (len(results1["states"])))

    print("\n--SOLUTION 1---")
    print_results(results1)

    print("\n--SOLUTION 2---")
    print_results(results2)

    print("\n\n------Solution 2 - Solution 1 Lengths-----")
    print_stats(lens2 - lens1, hist=False)
    print("%.2f%% soln2 equal to soln1" % (100 * np.mean(lens2 == lens1)))


if __name__ == "__main__":
    main()
