package main

func backtracing(board [][]string) bool {
	for i := 0; i < len(board); i++ { //row
		for j := 0; j < len(board[i]); j++ { //col
			if board[i][j] == "." {
				for k := '0'; k <= '9'; k++ {
					if isValid(i, j, k, board) {
						board[i][j] = string(k)
						result := backtracing(board)
						if result {
							return true
						}
						board[i][j] = "."
					}
				}
				return false
			}
		}
	}
	return true
}

func isValid(i int, j int, k rune, board [][]string) bool {
	return true
}
