from common_layer_naive import *

apple = 100
apple_num = 2
tax = 1.1

# Layer
mul_apple_layer = MulLayer()
mul_tax_layer = MulLayer()

# forward
apple_price = mul_apple_layer.forward(apple, apple_num)
price = mul_tax_layer.forward(apple_price, tax)

print(price)

# backword
dprice = 1
dapple_price, dtax = mul_tax_layer.backword(dprice)
dapple, dapple_num = mul_apple_layer.backword(dapple_price)

print(dapple, dapple_num, dtax)
