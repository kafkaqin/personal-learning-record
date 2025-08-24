### 01背包
### 完全背包
1. 重量 物品 价值 背包
2. 暴力解法 时间复杂度: 2的n次方 

### dp相关
1. dp[i][j] [0,i]物品任取放容量为j的背包的最大价值dp[i][j]
2. 不放物品i dp[i-1][j] ;放物品i： dp[i-1][j-weight]+value[i] ,max()
3. dp[i][j] = max(dp[i-1][j],dp[i-1][j-weight]+value[i])
4. 初始化: dp[i][0] = 0, dp[0][i] = 物品0的价值;非0下标初始化为0或者其他的都可以
![img.png](img.png)

```go
for () 物品
   for () 背包

```
![img_1.png](img_1.png)
