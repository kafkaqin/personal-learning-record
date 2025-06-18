package main

// 1. 客户支付5 直接收下
// 2. 客户支付10 找 5
// 3. 客户支付20 (10+5 或者 5+5+5)
func maxSum(bills []int) bool {
	var five int = 0
	var ten int = 0
	var twenty int = 0

	for _, bill := range bills {
		if bill == 5 {
			five++
		}
		if bill == ten {
			if five == 0 {
				return false
			}
			five--
			ten++
		}
		if bill == twenty {
			if ten > 0 && five > 0 {
				ten--
				five--
				twenty++
			} else if five >= 3 {
				five -= 3
				twenty++
			} else {
				return false
			}
		}
	}
	return true
}
