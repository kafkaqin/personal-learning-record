import numpy as np
import matplotlib.pyplot as plt

def demo:
    x = np.linspace(-30,30,500) # -30 30 的作用域

    y_sin = np.sin(x) # x 的正弦
    y_exp = np.exp(x) # x的指数
    y_cos = np.cos(x) # x的余弦
    plt.figure(figsize=(12,6))

    plt.subplot(1,3,1) # 分割窗口为1行3列，当前选中第一个
    plt.plot(x,y_sin,label='sin',color='blue')
    plt.title("Sine Function")
    plt.xlabel("x")
    plt.ylabel("sin(x)")
    plt.legend()

    plt.subplot(1,3,2) # 分割窗口为1行3列，当前选中第2个
    plt.plot(x,y_exp,label='exp',color='red')
    plt.title("Exponent Function")
    plt.xlabel("x")
    plt.ylabel("exp(x)")
    plt.legend()


    plt.subplot(1,3,3) # 分割窗口为1行3列，当前选中第3个
    plt.plot(x,y_cos,label='cos',color='green')
    plt.title("Cosine Function")
    plt.xlabel("x")
    plt.ylabel("cos(x)")
    plt.legend()

    plt.tight_layout()
    # plt.show()
    plt.savefig('test.png')

 if __name__ == '__main__':
     demo()