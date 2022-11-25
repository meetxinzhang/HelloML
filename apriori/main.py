# encoding: utf-8
"""This script calculate the Support of Apriori algorithm.

 @author: Xin Zhang
 @Student ID: 2250271011
 @contact: 2250271011@email.szu.edu.cn
 @time: 2022/11/25 13:52

 Typical usage example:

$ python main.py -i example_database.txt
$ python main.py -i example_database.txt -mins 3
$ python main.py -i example_database.txt -mins 3 -o output_file_path.txt

"""

import os
import argparse

parser = argparse.ArgumentParser(description='interface to run my Assignment 2: Apriori demo.')
parser.add_argument('-i', required=True, type=str, help=' Full local path of input, must be .txt file.')
parser.add_argument('-o', default='results.txt', type=str, help=' Full local path to output txt file.')
parser.add_argument('-mins', default=3, type=int, help=' The mini-support. default is 3')


def apriori(ts, mins, items):
    """Calculate the support of Apriori algorithm. My calculation involve only matrix addition and multipy, that can be
    accelerated by GPU in big data situation. Please Note that here I have to use many for loop because I can not use
    numpy or pandas.

    Args:
        ts: A list of transactions.
        mins: The mini-support. Integer.
    Returns:
        sups: A list which it's every element is the support of each transaction in trs list.
    """
    print('Input: \n', ts)
    print('Items: \n', items)

    # build multi-hot matrix, this pre-processing can be easy if I can import other package like numpy, pandas
    m = []
    for t in ts:
        # covert each transaction into an n-dim mask vector, in which 1 means the item exist while 0 is not
        t_vect = []
        for i in items:
            if i in t:
                t_vect.append(1)
            else:
                t_vect.append(0)
        m.append(t_vect)
    print('multi-hot matrix respect to input: \n', m)

    # get column vectors
    columns = [[row[column] for row in m] for column in range(len(items))]
    print('columns of matrix: \n', columns)

    # calculation details that involve only matrix operator
    # For 1-item subsets, the supports are:
    sups_1 = [sum(c) for c in columns if sum(c) >= mins]  # matrix addition in column dimension

    # for 2-item subset:
    


    print(sups_1)
    sups = []
    return sups


if __name__ == '__main__':
    # parse all parameters
    args = parser.parse_args()
    in_txt = args.i
    out_txt = args.o
    min_sup = args.mins

    # Robustness checking
    if min_sup < 0:
        exit(' Illegal input of -mins option, must be positive.')
    if not os.path.isfile(in_txt):
        exit(' Can not find the -i pointed file.')
    if not in_txt.endswith('.txt') or not out_txt.endswith('.txt'):
        exit(' Files pointed by -i and -o must be txt file.')
    if os.path.isfile(out_txt):
        exit(' The -o option pointed file is already exists.')

    transactions = []
    items = set()  # a set include all non-repeat item
    # Read transactions by lines from in_txt
    try:
        with open(in_txt) as f:
            for line in f:
                # transactions.append([int(e) for e in line.strip().split(' ')])
                new_line = []
                for e in line.strip().split(' '):
                    items.add(int(e))
                    new_line.append(int(e))
                transactions.append(new_line)
    except ValueError as e:
        print(' Non-integer exist in -i option file.')

    # Calculation
    supports = apriori(ts=transactions, mins=min_sup, items=items)

    # Write results into out_txt file
    # TODO

pass
