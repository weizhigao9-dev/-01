def is_safe(board, row, col):
    for i in range(row):
        # 检查列和对角线冲突
        if board[i] == col or abs(board[i] - col) == abs(i - row):
            return False
    return True

def solve_n_queens(n):
    results = []
    board = [-1] * n
    def backtrack(row):
        if row == n:
            results.append(list(board))
            return
        for col in range(n):
            if is_safe(board, row, col):
                board[row] = col
                backtrack(row + 1)
    backtrack(0)
    return results
