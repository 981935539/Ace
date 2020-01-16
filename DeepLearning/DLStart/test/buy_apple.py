# coding: utf-8
from DLStart.common import MulLayer, AddLayer


apple = 100
apple_num = 2
tax = 1.1

mul_apple_layer = MulLayer()
mul_tax_layer = MulLayer()

# forward
apple_price = mul_apple_layer.forward(apple, apple_num)
price = mul_tax_layer.forward(apple_price, tax)

print("apple * apple_num = {}".format(apple_price))
print("apple_price * tax = {}".format(int(price)))


# backward
dprice = 1
dapple_price, dtax = mul_tax_layer.backward(dprice)
dapple, dapple_num = mul_apple_layer.backward(dapple_price)

print("dprice:", dprice)
print("dapple_price:", dapple_price)
print("dtax:", dtax)
print("dApple:", dapple)
print("dapple_num:", int(dapple_num))

