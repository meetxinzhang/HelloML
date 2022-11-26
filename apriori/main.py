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
from itertools import combinations

parser = argparse.ArgumentParser(description='interface to run my Assignment 2: Apriori demo.')
parser.add_argument('-i', required=True, type=str, help=' Full local path of input, must be .txt file.')
parser.add_argument('-o', default='results.txt', type=str, help=' Full local path to output txt file.')
parser.add_argument('-mins', default=3, type=int, help=' The mini-support. default=3')


def apriori_xz(tsc: list[list[int]], mins: int, items: list[int]):
    """Calculate the support of Apriori algorithm. My calculation involve only matrix addition and multipy, that can be
    accelerated by GPU in big data situation. Please Note that here I have to use many for loop because I can not use
    numpy or pandas.

    Args:
        tsc: A list of transactions, each transaction is also a list.
        mins: The mini-support. Integer.
        items: A list contains all appeared items
    Returns:
        sups: A list which it's every element is the support of each transaction in trs list.
    """

    columns = len(items)
    rows = len(tsc)
    # build mask matrix, this pre-processing can be simplified if I can import numpy or pandas
    column_vectors = [[0]*rows for _ in range(columns)]  # new the initial full 0 mask matrix=[columns, rows]
    for c, i in zip(range(columns), items):
        for r, t in zip(range(rows), tsc):
            if i in t:   # if the Item i is existing in Transaction t
                column_vectors[c][r] = 1  # put 1 at [c, r] site of matrix
    print('\ncolumns of matrix: \n', column_vectors)

    # Calculation details !!!!!!!!!!!!!!!!!
    # For 1-item subsets, just do addition in column dimension.
    # The interest subsets are those whose sum(column vector)>mins.
    answer = {}  # dict for output
    idx_1 = []  # list for index of interest single item.
    for idx in range(columns):
        if sum(column_vectors[idx]) >= mins:
            idx_1.append(idx)
            answer[str(items[idx])] = sum(column_vectors[idx])

    # for n-item subsets, combine the 1-item subsets, then do matrix multipy.
    # The interest subsets are those subsets whose (sum of product of column vectors)>mins
    # Here I do dot-product and addition by for loop since no numpy. The two operators can compose into matrix multipy.
    for n in range(2, len(idx_1)+1):
        for com in combinations(idx_1, n):
            product = [1] * rows
            for idx in com:
                product = list(map(lambda x, y: x * y, product, column_vectors[idx]))  # dot-product
            if sum(product) >= mins:  # addition
                answer[str([items[i] for i in com]).replace('[', '').replace(']', '').replace(',', ' ')] = sum(product)

    return answer


if __name__ == '__main__':
    # parse all parameters
    args = parser.parse_args()
    in_txt = args.i
    out_txt = args.o
    min_sup = args.mins

    # Robustness checking
    if min_sup <= 0:
        exit(' Illegal input of -mins option, must be positive.')
    if not os.path.isfile(in_txt):
        exit(' Can not find the -i pointed file.')
    if not in_txt.endswith('.txt') or not out_txt.endswith('.txt'):
        exit(' Files pointed by -i and -o must be txt file.')
    if os.path.isfile(out_txt):
        exit(' The -o option pointed file is already exists.')

    # Read transactions by lines from in_txt
    transactions = []
    items = set()  # a set include all non-repeat items
    try:
        with open(in_txt) as f:
            for line in f:
                new_line = []
                for e in line.strip().split(' '):
                    items.add(int(e))
                    new_line.append(int(e))
                transactions.append(new_line)
    except ValueError as e:
        print(' Non-integer elements exist in -i option pointed file.')
    items = list(items)
    items.sort()
    print('\nInput: \n', transactions)
    print('\nItems: \n', items)

    # Calculation
    answer = apriori_xz(tsc=transactions, mins=min_sup, items=items)

    # Write answer into out_txt file
    print('\nResults for -min_sup '+str(min_sup)+':')
    with open(out_txt, 'w') as f:
        for key, val in answer.items():
            f.write(key + ' #SUP: ' + str(val) + '\n')
            print(key + ' #SUP: ' + str(val))
    print('\nSaved answer in '+out_txt)

pass
