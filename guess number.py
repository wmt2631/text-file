# tisi is guess number game!
# 2018/2/8编写有点改进！！
import random

number = random.randint(1, 20)
print("输入一个1-20的数字: ")
# 随机一个1-20的数
a = "你猜错了，亲重新输入,"
b = "你猜的过大！"
c = "你猜的过小！"
time = 5
guess = int(input())

while (guess != number) and (time > 0):
    if guess >= 20:
        print("你输入的数字大于20")
    else:
        # print(a)
        if guess == number:
            print("你真的太聪明了，你猜对了！")
            break
        else:
            # 大小
            if guess < number:
                print(a + c)
            else:
                print(a + b)
        guess = int(input())
        time = time - 1
        if time > 1:
            print("请再次输入!")
        else:
            print("次数已用完，你没有猜对！")
print("你真的太聪明了，你猜对了！")
