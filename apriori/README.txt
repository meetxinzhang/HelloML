# encoding: utf-8

Document for assignment 2: Apriori

 @author: Xin Zhang
 @Student ID: 2250271011
 @contact: 2250271011@email.szu.edu.cn
 @time: 2022/11/25 13:52

 Typical usage example:

 $ python main.py -i example_database.txt
 $ python main.py -i example_database.txt -mins 3
 $ python main.py -i example_database.txt -mins 3 -o output_file_path.txt

 -i:    Required, string. Full local path of input, must be txt file.
 -mins: Default=3, positive int. The mini-support. default=3
 -o:    Default='results.txt', string. Full local path of a txt file to output. Can ont be an already existing file.


Core idea:
 Different to other method, my algorithm covert the Support calculation into MATRIX multiplication and addition.
  In this way the calculation can be accelerated by GPU which is useful in big data.
  Parallel run is also supported by matrix decomposition (not online yet).


Algorithm details:
 First, Regard the transactions as a mask matrix, let's take the following input as example:

 >>input
 1 3 4
 2 3 5
 1 2 3 5
 2 5
 1 2 3 5

 >>all appeared items, sorted by number.
 1 2 3 4 5

 >>mask matrix

 0 1 2 3 4: columns index
 1 2 3 4 5: items value

 1 0 1 1 0
 0 1 1 0 1
 1 1 1 0 1
 0 1 0 0 1
 1 1 1 0 1


 where, rows are transactions and columns are items. We know the item values are 1 2 3 4 5, let (0 1 2 3 4) are the index
 of columns for directly description. matrix element value 1 indicate an item appeared in a transaction while 0 is not.
 e.g. the first row [1 0 1 1 0] means (0 2 3) indexed items are appeared. (0 2 3) indexed items are num 1 3 4.

 Second, calculate all supports of single-item subsets by just addition all columns, obviously.
 >>sum(columns)

 0 1 2 3 4: columns index
 1 2 3 4 5: items value

 3 4 4 1 4

 Let the min_sup is 3, and we get the interest single-item subsets
 >>single-item subsets
 1 #SUP: 3
 2 #SUP: 4
 3 #SUP: 4
 5 #SUP: 4

 For multi-item subsets, do Hadamard Product (element-wise product) between column vectors in each subset. Then do column
  addition like before. To do this, we need to find the full-combination of all above single-item subsets. Their index is
  (0 1 2 4).
 >>combine(index: 0 1 2 4)
 [0 1], [0 2], [0 4], [1 2], [1 4], [2 4], [0 1 2], [0 1 4], [0 2 4], [1 2 4], [0 1 2 4]

 Finally, perform Hadamard Product between column vectors for each above subset. Let's take (1 2 4) as example:
 >>mask matrix -> extract columns (1 2 4)

 0 1 2 3 4: columns index
 1 2 3 4 5: items value

   0 1   0
   1 1   1
   1 1   1
   1 0   1
   1 1   1

 >>Hadamard Product(dim=column)
 0
 1
 1
 0
 1

 >>sum()
 3

 where 3 >= min_sup is True, so the [1 2 4] subset is one of the answer.


Discussion:
 As you can see, the algorithm involves only column addition and Hadamard multiplication in a mask matrix. The two operator
 can compose into matrix multiplication when the columns are even number.

 I am not sure if I can use numpy or pandas package here to make the coding convenient and easy to read. In my submitted
 version no extra package import, just Python3 standard library. Since this I have to write many for loop to do matrix operator.

Online version:
https://github.com/MeetXinZhang/HelloML/tree/master/apriori
