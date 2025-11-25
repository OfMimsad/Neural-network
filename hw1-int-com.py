# MohammadReza Safarpour 404464116
# Aritificail Computation:  First Homework


x_inputs = [[0,0],
            [0,1],
            [1,0],
            [1,1]]

t_and =[0,0,0,1]
t_or = [0,1,1,1]

lr = 1 #learning rate
b1 = 0
w1 = 0
w3 = 0
w2 = 0
w4 = 0
b2 = 0


def sign_function(net):
    if net > 0:
        return 1
    else:
        return 0


#*************************  AND  ****************************
for epoch in range(10):
    for i in range(len(x_inputs)):
        x1,x2 = x_inputs[i][0] , x_inputs[i][1]
        net = x1*w1 + x2*w3 + b1
        y = sign_function(net)
        error = t_and[i] - y
        if error != 0 :
            w1 = w1 + lr*x1*error
            w3 = w3 + lr*x2*error
            b1 = b1 + lr*error
 
Y1 = []
for i in range(len(x_inputs)):
    x1,x2 = x_inputs[i][0] , x_inputs[i][1]
    net = x1*w1 + x2*w3 + b1
    Y1.append(sign_function(net))
#*************************************************************



#**** OR ****
for epoch in range(20):
    for i in range(len(x_inputs)):
        x1, x2 = x_inputs[i][0], x_inputs[i][1]
        net = x1*w2 + x2*w4 + b2
        y = sign_function(net)
        error = t_or[i] - y
        if error != 0:
            w2 = w2 + lr*x1*error
            w4 = w4 + lr*x2*error
            b2 = b2 + lr*error



Y2 = []
for i in range(len(x_inputs)):
    x1,x2 = x_inputs[i][0] , x_inputs[i][1]
    net = x1*w2 + x2*w4 + b2 
    Y2.append(sign_function(net))
 #**********************************************



#***** Z and *****
w5 = 0
w6 = 0
b3 = 0
t_z = [0,1,1,0]
for epoch in range(10):
    for i in range(len(Y1)):
        y1 = Y1[i]
        y2 = Y2[i]
        net = y1*w5 + y2*w6 + b3
        z = sign_function(net)
        error = t_z[i] - z
        if error != 0:
            w5 = w5 + lr*y1*error
            w6 = w6 + lr*y2*error
            b3 = b3 + lr*error


Z = []
for i in range(len(Y1)):
        y1 = Y1[i]
        y2 = Y2[i] 
        net = y1*w5 + y2*w6 + b3
        Z.append(sign_function(net))


print(Z)