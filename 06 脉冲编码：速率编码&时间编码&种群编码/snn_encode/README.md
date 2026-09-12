# SNN 脉冲编码实验室

一个无需安装依赖的交互式教学页面，基于 `snn_encoders.py` 中的编码逻辑重新实现为浏览器端 JavaScript。

## 使用方法

直接双击打开 [index.html](index.html)，或在本目录启动任意静态文件服务后访问页面。例如：

```powershell
python -m http.server 8000
```

然后在浏览器打开 `http://localhost:8000`。

## 包含的编码方式

- 确定性速率、泊松、延迟（TTFS）、相位、二进制时间编码
- 高斯群体编码，以及群体结合确定性／随机脉冲编码
- Gaussian tuning + latency 编码
- 脉冲顺序（Rank-order）、三脉冲 ISI Pattern、突发（Burst）、Burst + ISI、事件／差分编码

页面会按结果类型显示二值脉冲栅格或连续群体响应热力图；脉冲栅格每行最右侧显示对应的发放率。
