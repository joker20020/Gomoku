#include "gomoku.h"

// ���캯��
GomokuBoard::GomokuBoard() {
    InitializeBoard();
    currentPlayer = BLACK;  // ��������
}

// ��������
GomokuBoard::~GomokuBoard() {
    // �������⴦��
}

// ��ʼ������
void GomokuBoard::InitializeBoard() {
    board = vector<vector<Color>>(BOARD_SIZE, vector<Color>(BOARD_SIZE, EMPTY));
}

// ��ӡ����
void GomokuBoard::PrintBoard() const {
    // ��ӡ�б�ǩ�����֣���0��ʼ��
    cout << "   ";  // �����б�ǩ
    for (int col = 0; col < BOARD_SIZE; ++col) {
        cout << " " << setw(2) << col << " ";  // �б�ǩ��0-14�����̶�����Ϊ2
    }
    cout << endl;

    // ��ӡ��������
    for (int row = 0; row < BOARD_SIZE; ++row) {
        // ��ӡ�б�ǩ�����֣���0��ʼ��
        cout << setw(2) << row << " ";  // �б�ǩ��0-14�����̶�����Ϊ2

        // ��ӡ��������
        for (int col = 0; col < BOARD_SIZE; ++col) {
            switch (board[row][col]) {
            case EMPTY:
                cout << " +  ";  // �ո��ʾ��λ
                break;
            case BLACK:
                cout << " \033[1;31mB\033[0m  ";  // ���� red
                break;
            case WHITE:
                cout << " \033[1;33mW\033[0m  ";  // ���� yellow
                break;
            }
        }
        cout << " " << setw(2) << row << endl;  // �б�ǩ��0-14�����̶�����Ϊ2
    }

    // ��ӡ�б�ǩ�����֣���0��ʼ��
    cout << "   ";  // �����б�ǩ
    for (int col = 0; col < BOARD_SIZE; ++col) {
        cout << " " << setw(2) << col << " ";  // �б�ǩ��0-14�����̶�����Ϊ2
    }
    cout << endl;
}

// ����
bool GomokuBoard::PlacePiece(int row, int col, Color color) {
    if (!IsValidMove(row, col)) return false;
    board[row][col] = color;
    return true;
}

Color GomokuBoard::GetPiece(int row, int col) const{
    return board[row][col];
}

// �ж������Ƿ�Ϸ�
bool GomokuBoard::IsValidMove(int row, int col) const {
    if (row < 0 || row >= BOARD_SIZE || col < 0 || col >= BOARD_SIZE) return false;
    return board[row][col] == EMPTY;
}

// �жϵ�ǰ�����Ƿ��γ���������
bool GomokuBoard::CheckWin(int row, int col, Color color) const {
    // ����ĸ�����ˮƽ����ֱ���Խ��ߣ����ϵ����£����Խ��ߣ����ϵ����£�
    int directions[4][2] = { {1, 0}, {0, 1}, {1, 1}, {1, -1} };

    for (auto dir : directions) {
        int count = 1;  // ��ǰ����
        int dx = dir[0], dy = dir[1];

        // ��һ����������
        for (int i = 1; i < 5; ++i) {
            int newRow = row + i * dx;
            int newCol = col + i * dy;
            if (newRow < 0 || newRow >= BOARD_SIZE || newCol < 0 || newCol >= BOARD_SIZE) break;
            if (board[newRow][newCol] != color) break;
            count++;
        }

        // ���෴��������
        for (int i = 1; i < 5; ++i) {
            int newRow = row - i * dx;
            int newCol = col - i * dy;
            if (newRow < 0 || newRow >= BOARD_SIZE || newCol < 0 || newCol >= BOARD_SIZE) break;
            if (board[newRow][newCol] != color) break;
            count++;
        }

        if (count >= 5) return true;  // ��������
    }

    return false;
}

// ��ȡ��ǰ���
Color GomokuBoard::GetCurrentPlayer() const {
    return currentPlayer;
}

// �л����
void GomokuBoard::SwitchPlayer() {
    currentPlayer = (currentPlayer == BLACK) ? WHITE : BLACK;
}

// �ж���Ϸ�Ƿ����������⵱ǰ����λ�ã�
GameResult GomokuBoard::IsGameOver(int row, int col) const {
    Color color = board[row][col];
    if (color == EMPTY) return NOT_OVER;  // �����ǰλ��Ϊ�գ���Ϸδ����

    // ��鵱ǰ�����Ƿ��γ���������
    if (CheckWin(row, col, color)) {
        return (color == BLACK) ? BLACK_WIN : WHITE_WIN;
    }

    // ��������Ƿ�������ƽ�֣�
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

    return NOT_OVER;  // ��Ϸδ����
}

// ��������״̬
double GomokuBoard::EvaluateBoard(GameResult result, Color currentPlayer) const {
    if (result == BLACK_WIN && currentPlayer == BLACK) return -1.0;
    else if (result == WHITE_WIN && currentPlayer == WHITE) return -1.0;
    else if (result == BLACK_WIN && currentPlayer == WHITE) return 1.0;
    else if (result == WHITE_WIN && currentPlayer == BLACK) return 1.0;
    else if (result == DRAW) return 0.0;
    else return 0.0;  // ��Ϸδ����������0
}

// ���ɺϷ��ƶ�
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

// ����
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

torch::Tensor RlGomokuBoard::DumpBoard(bool swap) {
    torch::Tensor dumpBoard = torch::zeros({ LAST_NUM * 2, BOARD_SIZE, BOARD_SIZE});
    torch::Tensor currentBoard = torch::zeros({2, BOARD_SIZE, BOARD_SIZE});
    pair<pair<int, int>, Color> move;
    for (size_t row = 0; row < BOARD_SIZE; row++) {
        for (size_t col = 0; col < BOARD_SIZE; col++)
        {
            if (swap) {
                if (board[row][col] == BLACK)
                {
                    currentBoard[1][row][col] = 1;
                }
                else if (board[row][col] == WHITE)
                {
                    currentBoard[0][row][col] = 1;
                }
            }
            else {
                if (board[row][col] == BLACK)
                {
                    currentBoard[0][row][col] = 1;
                }
                else if (board[row][col] == WHITE)
                {
                    currentBoard[1][row][col] = 1;
                }
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