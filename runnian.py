year = int(input())
a = ":是闰年"
b = ":不是闰年"
while not year.isdigit():
    if year % 4 == 0:
        print('year' + a)
    else:
        print('year' + b)
