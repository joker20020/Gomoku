#include "gomoku.h"

// 构造函数
GomokuBoard::GomokuBoard() {
    InitializeBoard();
    currentPlayer = BLACK;  // 黑棋先手
}

// 析构函数
GomokuBoard::~GomokuBoard() {
    // 无需特殊处理
}

// 初始化棋盘
void GomokuBoard::InitializeBoard() {
    board = vector<vector<Color>>(BOARD_SIZE, vector<Color>(BOARD_SIZE, EMPTY));
}

// 打印棋盘
void GomokuBoard::PrintBoard() const {
    // 打印列标签（数字，从0开始）
    cout << "   ";  // 对齐行标签
    for (int col = 0; col < BOARD_SIZE; ++col) {
        cout << " " << setw(2) << col << " ";  // 列标签（0-14），固定宽度为2
    }
    cout << endl;

    // 打印棋盘内容
    for (int row = 0; row < BOARD_SIZE; ++row) {
        // 打印行标签（数字，从0开始）
        cout << setw(2) << row << " ";  // 行标签（0-14），固定宽度为2

        // 打印棋盘内容
        for (int col = 0; col < BOARD_SIZE; ++col) {
            switch (board[row][col]) {
            case EMPTY:
                cout << " +  ";  // 空格表示空位
                break;
            case BLACK:
                cout << " B  ";  // 黑棋
                break;
            case WHITE:
                cout << " W  ";  // 白棋
                break;
            }
        }
        cout << " " << setw(2) << row << endl;  // 行标签（0-14），固定宽度为2
    }

    // 打印列标签（数字，从0开始）
    cout << "   ";  // 对齐行标签
    for (int col = 0; col < BOARD_SIZE; ++col) {
        cout << " " << setw(2) << col << " ";  // 列标签（0-14），固定宽度为2
    }
    cout << endl;
}

// 落子
bool GomokuBoard::PlacePiece(int row, int col, Color color) {
    if (!IsValidMove(row, col)) return false;
    board[row][col] = color;
    return true;
}

Color GomokuBoard::GetPiece(int row, int col) const{
    return board[row][col];
}

// 判断落子是否合法
bool GomokuBoard::IsValidMove(int row, int col) const {
    if (row < 0 || row >= BOARD_SIZE || col < 0 || col >= BOARD_SIZE) return false;
    return board[row][col] == EMPTY;
}

// 判断当前落子是否形成五子连珠
bool GomokuBoard::CheckWin(int row, int col, Color color) const {
    // 检查四个方向：水平、垂直、对角线（左上到右下）、对角线（右上到左下）
    int directions[4][2] = { {1, 0}, {0, 1}, {1, 1}, {1, -1} };

    for (auto dir : directions) {
        int count = 1;  // 当前棋子
        int dx = dir[0], dy = dir[1];

        // 向一个方向搜索
        for (int i = 1; i < 5; ++i) {
            int newRow = row + i * dx;
            int newCol = col + i * dy;
            if (newRow < 0 || newRow >= BOARD_SIZE || newCol < 0 || newCol >= BOARD_SIZE) break;
            if (board[newRow][newCol] != color) break;
            count++;
        }

        // 向相反方向搜索
        for (int i = 1; i < 5; ++i) {
            int newRow = row - i * dx;
            int newCol = col - i * dy;
            if (newRow < 0 || newRow >= BOARD_SIZE || newCol < 0 || newCol >= BOARD_SIZE) break;
            if (board[newRow][newCol] != color) break;
            count++;
        }

        if (count >= 5) return true;  // 五子连珠
    }

    return false;
}

// 获取当前玩家
Color GomokuBoard::GetCurrentPlayer() const {
    return currentPlayer;
}

// 切换玩家
void GomokuBoard::SwitchPlayer() {
    currentPlayer = (currentPlayer == BLACK) ? WHITE : BLACK;
}

// 判断游戏是否结束（仅检测当前落子位置）
GameResult GomokuBoard::IsGameOver(int row, int col) const {
    Color color = board[row][col];
    if (color == EMPTY) return NOT_OVER;  // 如果当前位置为空，游戏未结束

    // 检查当前落子是否形成五子连珠
    if (CheckWin(row, col, color)) {
        return (color == BLACK) ? BLACK_WIN : WHITE_WIN;
    }

    // 检查棋盘是否已满（平局）
    bool isBoardFull = true;
    for (int row = 0; row < BOARD_SIZE; ++row) {
        for (int col = 0; col < BOARD_SIZE; ++col) {
            if (board[row][col] == EMPTY) {
                isBoardFull = false;
                break;
            }
        }
        if (!isBoardFull) break;
    }

    if (isBoardFull) return DRAW;

    return NOT_OVER;  // 游戏未结束
}

// 评估棋盘状态
double GomokuBoard::EvaluateBoard(GameResult result, Color currentPlayer) const {
    if (result == BLACK_WIN && currentPlayer == BLACK) return -1.0;
    else if (result == WHITE_WIN && currentPlayer == WHITE) return -1.0;
    else if (result == BLACK_WIN && currentPlayer == WHITE) return 1.0;
    else if (result == WHITE_WIN && currentPlayer == BLACK) return 1.0;
    else if (result == DRAW) return 0.0;
    else return 0.0;  // 游戏未结束，返回0
}

// 生成合法移动
vector<pair<int, int>> GomokuBoard::GenerateLegalMoves(const GomokuBoard& board, Color player){
    vector<pair<int, int>> moves;
    for (int row = 0; row < BOARD_SIZE; ++row) {
        for (int col = 0; col < BOARD_SIZE; ++col) {
            if (board.GetPiece(row, col) == EMPTY) {
                moves.push_back({row, col});
            }
        }
    }
    return moves;
}


RlGomokuBoard::RlGomokuBoard():GomokuBoard() {
}

// 落子
bool RlGomokuBoard::PlacePiece(int row, int col, Color color) {
    if (GomokuBoard::PlacePiece(row, col, color))
    {
        lastMoves.push_back({ {row, col}, color });
        if (lastMoves.size() > LAST_NUM)
        {
            lastMoves.erase(lastMoves.begin());
        }
        return true;
    }
    return false;
}

torch::Tensor RlGomokuBoard::DumpBoard() {
    torch::Tensor dumpBoard = torch::zeros({ LAST_NUM * 2, BOARD_SIZE, BOARD_SIZE});
    torch::Tensor currentBoard = torch::zeros({2, BOARD_SIZE, BOARD_SIZE});
    pair<pair<int, int>, Color> move;
    for (size_t row = 0; row < BOARD_SIZE; row++) {
        for (size_t col = 0; col < BOARD_SIZE; col++)
        {
            if (board[row][col] == BLACK)
            {
                currentBoard[0][row][col] = 1;
            }
            else if(board[row][col] == WHITE)
            {
                currentBoard[1][row][col] = 1;
            }
        }
    }
    vector<pair<pair<int, int>, Color>>::reverse_iterator iter = lastMoves.rbegin();
    for (; iter != lastMoves.rend(); iter++)
    {
        move = *iter;
        dumpBoard = torch::cat({ 
            dumpBoard.index({Slice(-(LAST_NUM - 1) * 2,None)}),
            currentBoard
            });
        if (move.second == BLACK)
        {
            currentBoard[0][move.first.first][move.first.second] = 0;
        }
        else if (move.second == WHITE) {
            currentBoard[1][move.first.first][move.first.second] = 0;
        }
    }
    return dumpBoard;
}