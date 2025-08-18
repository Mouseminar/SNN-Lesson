import torch
import matplotlib.pyplot as plt

# 参数设置
tau = 2.0
T = 100
v_threshold = 1.0
v_reset_val = 0.2  # 非零 reset 情况
torch.manual_seed(42)
X = torch.rand(T) * 1.2
# X = torch.ones(T) * 0.4

# 初始化电位与脉冲记录
U1 = torch.zeros(T+1); S1 = torch.zeros(T)
U2 = torch.zeros(T+1); S2 = torch.zeros(T)
U3 = torch.zeros(T+1); S3 = torch.zeros(T)
U4 = torch.zeros(T+1); S4 = torch.zeros(T)

# 情况① decay_input=True, v_reset=0
for t in range(1, T+1):
    U1[t] = U1[t-1] + (X[t-1] - U1[t-1]) / tau
    if U1[t] >= v_threshold:
        S1[t-1] = 1
        U1[t] = 0.0  # reset to v_reset=0

# 情况② decay_input=True, v_reset=非零
for t in range(1, T+1):
    U2[t] = U2[t-1] + (X[t-1] - (U2[t-1] - v_reset_val)) / tau
    if U2[t] >= v_threshold:
        S2[t-1] = 1
        U2[t] = v_reset_val

# 情况③ decay_input=False, v_reset=0
for t in range(1, T+1):
    U3[t] = U3[t-1] * (1 - 1 / tau) + X[t-1]
    if U3[t] >= v_threshold:
        S3[t-1] = 1
        U3[t] = 0.0

# 情况④ decay_input=False, v_reset=非零
for t in range(1, T+1):
    U4[t] = U4[t-1] - (U4[t-1] - v_reset_val) / tau + X[t-1]
    if U4[t] >= v_threshold:
        S4[t-1] = 1
        U4[t] = v_reset_val

# 去掉 U[0]
U1 = U1[1:]
U2 = U2[1:]
U3 = U3[1:]
U4 = U4[1:]

# 绘图
plt.figure(figsize=(12, 10))

plt.subplot(5,1,1)
plt.plot(X.numpy())
plt.title("Input X[t]")

plt.subplot(5,1,2)
plt.plot(U1.numpy(), label="U1")
plt.stem(range(T), S1.numpy(), linefmt='r-', markerfmt='ro', basefmt='k')
plt.title("① decay_input=True, v_reset=0")

plt.subplot(5,1,3)
plt.plot(U2.numpy(), label="U2")
plt.stem(range(T), S2.numpy(), linefmt='r-', markerfmt='ro', basefmt='k')
plt.title("② decay_input=True, v_reset=0.2")

plt.subplot(5,1,4)
plt.plot(U3.numpy(), label="U3")
plt.stem(range(T), S3.numpy(), linefmt='r-', markerfmt='ro', basefmt='k')
plt.title("③ decay_input=False, v_reset=0")

plt.subplot(5,1,5)
plt.plot(U4.numpy(), label="U4")
plt.stem(range(T), S4.numpy(), linefmt='r-', markerfmt='ro', basefmt='k')
plt.title("④ decay_input=False, v_reset=0.2")

plt.tight_layout()
plt.show()
