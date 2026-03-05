import sys
import os
# 关键：让 Python 能在 GitHub 的路径下找到 src 文件夹
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from queens import solve_n_queens

def test_answer():
    assert len(solve_n_queens(4)) == 2
    assert len(solve_n_queens(8)) == 92
    print("Test Passed!")

if __name__ == "__main__":
    test_answer()
