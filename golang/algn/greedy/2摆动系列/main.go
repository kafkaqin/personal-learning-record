package main

// 一正一负 pre_diff >= 0 && cur_diff <0
// 一正一负 pre_diff <= 0 && cur_diff >0
// 1.上下坡有平坡
// 2.首尾元素
// 3.单调坡中有平坡
func greety(nums []int) int {
	result := 1
	if len(nums) == 1 {
		return result
	}
	prediff := 0
	curdiff := 0
	for i := 0; i < len(nums)-1; i++ {
		curdiff = nums[i+1] - nums[i]
		if (prediff >= 0 && curdiff < 0) || (prediff <= 0 && curdiff > 0) {
			result++
			prediff = curdiff
		}
	}
	return result
}
