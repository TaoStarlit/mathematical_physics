import math

z = [1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0]
print("z vecter: {}".format(z))
z_exp = [math.exp(i) for i in z]
print("e raised to zth power: {}".format([round(i, 2) for i in z_exp]))#if no []  round for i in list    is just a object, not list

sum_z_exp = sum(z_exp)
print("sum_z_exponential: {:.2f}".format(sum_z_exp)) #{:.2f} has the same usage with round(x,2)

softmax = [round(i / sum_z_exp, 3) for i in z_exp]

print("softmax: {}".format(softmax))
print("sum of softmax: {}".format(sum(softmax)))
