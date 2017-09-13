#coding=utf-8
list_color = raw_input('input the hex=')

red_hex = list_color[:2]
green_hex = list_color[2:4]
blue_hex = list_color[4:]

red_dec = int(red_hex,16)
green_dec = int(green_hex,16)
blue_dec = int(blue_hex,16)

print'red=%d'%red_dec
print'green=%d'%green_dec
print'blue=%d'%blue_dec