package main

// dp[i][j] 以i-1为结尾的nums1[i]，以j-1为结尾的nums2[j]最长重复子数组
// if num1[i-1] > nums2[j-1] {dp[i][j] = dp[i-1][j-1]+1}
func getMaxLength(nums1, nums2 []int) int {
	lenNums1 := len(nums1)
	lenNums2 := len(nums2)
	dp := make([][]int, lenNums1+1)
	dp[0][0] = 0
	result := 0
	for i := 1; i <= lenNums1; i++ {
		for j := 1; j <= lenNums2; j++ {
			if nums1[i-1] > nums2[j-1] {
				dp[i][j] = dp[i-1][j-1] + 1
				result = max(result, dp[i][j])
			}

		}
	}
	return result
}

func max(a, b int) int {
	if a > b {
		return a
	}
	return b
}
