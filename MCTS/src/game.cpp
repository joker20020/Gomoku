#include "game.h"

ChessGame::ChessGame() {
    this->currentPlayer = BLACK;
    this->ai = new MCTSAI();
}

ChessGame::ChessGame(GomokuBoard* board, Color aiColor) {
    this->currentPlayer = BLACK;
    this->ai = new MCTSAI(*board, aiColor);
    this->aiColor = ai->root->currentPlayer;
    this->board = board;
}

ChessGame::~ChessGame(){
    delete ai;
}

void ChessGame::Start() {
    time_t startTime, endTime;
    int row, col;
    bool gameOver = false;
    pair<int, int> bestMove;
    while (!gameOver) {
        board->PrintBoard();
        currentPlayer = board->GetCurrentPlayer();
        
        if (currentPlayer == aiColor) {
            time(&startTime);
            ai->ParallelRun(2000);
            time(&endTime);
            cout << "ai use time: " << (difftime(endTime, startTime)) << "s" << endl;
            bestMove = ai->GetBestMove();
            row = bestMove.first;
            col = bestMove.second;
            cout << (currentPlayer == BLACK ? "Black" : "White") << "'s turn. ai move: " << row << " " << col << endl;
        }
        else {
            cout << (currentPlayer == BLACK ? "Black" : "White") << "'s turn. Enter row and column (e.g., 8 8): " << endl;
            cin >> row >> col;
            bestMove.first = row;
            bestMove.second = col;
        }

        if (board->PlacePiece(row, col, currentPlayer)) {
            GameResult result = board->IsGameOver(row, col);
            ai->Update(bestMove);
            /*cout << "ai updated" << endl;*/
            if (result != NOT_OVER) {
                board->PrintBoard();
                if (result == BLACK_WIN) {
                    cout << "Black wins!" << endl;
                }
                else if (result == WHITE_WIN) {
                    cout << "White wins!" << endl;
                }
                else {
                    cout << "It's a draw!" << endl;
                }
                gameOver = true;
            }
            else {
                board->SwitchPlayer();
            }
        }
        else {
            cout << "Invalid move. Try again." << endl;
        }
    }
}

vector<pair<int, int>> ChessGame::ParseInput(const string& input) {
    vector<pair<int, int>> positions;

    // 检查输入格式
    if (input.length() != 5 || input[2] != ' ') {
        return positions; // 返回空列表表示输入无效
    }

    // 解析起始位置
    char startColChar = input[0];
    char startRowChar = input[1];
    if (startColChar < 'a' || startColChar > 'i' || startRowChar < '0' || startRowChar > '9') {
        return positions; // 返回空列表表示输入无效
    }
    int startCol = startColChar - 'a';
    int startRow = startRowChar - '0';

    // 解析目标位置
    char endColChar = input[3];
    char endRowChar = input[4];
    if (endColChar < 'a' || endColChar > 'i' || endRowChar < '0' || endRowChar > '9') {
        return positions; // 返回空列表表示输入无效
    }
    int endCol = endColChar - 'a';
    int endRow = endRowChar - '0';

    // 返回解析后的坐标
    positions.push_back({startRow, startCol});
    positions.push_back({endRow, endCol});
    return positions;
}

RlChessGame::RlChessGame() {
    this->currentPlayer = BLACK;
}

RlChessGame::RlChessGame(shared_ptr<RlGomokuBoard> board, Color aiColor, shared_ptr<MCTSModelPool> modelPool) {
    this->currentPlayer = BLACK;
    this->ai = make_shared<RlMCTSAI>(*board, aiColor, modelPool);
    this->aiColor = ai->root->currentPlayer;
    this->board = board;
}

RlChessGame::~RlChessGame() {
    
}

void RlChessGame::Start() {
    time_t startTime, endTime;
    int row, col;
    bool gameOver = false;
    pair<int, int> bestMove;
    while (!gameOver) {
        board->PrintBoard();
        currentPlayer = board->GetCurrentPlayer();

        if (currentPlayer == aiColor) {
            time(&startTime);
            ai->ParallelRun(1000);
            time(&endTime);
            cout << "ai use time: " << (difftime(endTime, startTime)) << "s" << endl;
            bestMove = ai->GetBestMove();
            row = bestMove.first;
            col = bestMove.second;
            cout << (currentPlayer == BLACK ? "Black" : "White") << "'s turn. ai move: " << row << " " << col << endl;
        }
        else {
            cout << (currentPlayer == BLACK ? "Black" : "White") << "'s turn. Enter row and column (e.g., 8 8): " << endl;
            cin >> row >> col;
            bestMove.first = row;
            bestMove.second = col;
        }

        if (board->PlacePiece(row, col, currentPlayer)) {
            GameResult result = board->IsGameOver(row, col);
            ai->Update(bestMove);
            /*cout << "ai updated" << endl;*/
            if (result != NOT_OVER) {
                board->PrintBoard();
                if (result == BLACK_WIN) {
                    cout << "Black wins!" << endl;
                }
                else if (result == WHITE_WIN) {
                    cout << "White wins!" << endl;
                }
                else {
                    cout << "It's a draw!" << endl;
                }
                gameOver = true;
            }
            else {
                board->SwitchPlayer();
            }
        }
        else {
            cout << "Invalid move. Try again." << endl;
        }
    }
}

pair<torch::Tensor, pair<torch::Tensor, torch::Tensor>> RlChessGame::TrainStart() {
    time_t startTime, endTime;
    int row, col;
    bool gameOver = false;
    vector<Color> history = {};
    torch::Tensor boardTensor, pTensor, vTensor, pTemp;
    GameResult result;
    size_t turn = 0;
    
    boardTensor = torch::cat({ torch::zeros({1, BOARD_SIZE, BOARD_SIZE}),board->DumpBoard() }).unsqueeze(0);
    
    history.push_back(board->GetCurrentPlayer());

    pair<int, int> bestMove;

    while (!gameOver) {
        board->PrintBoard();
        currentPlayer = board->GetCurrentPlayer();

        time(&startTime);
        ai->ParallelRun(10, 1);
        time(&endTime);
        cout << "ai use time: " << (difftime(endTime, startTime)) << "s" << endl;
        bestMove = ai->GetBestMove();
        row = bestMove.first;
        col = bestMove.second;
        cout << (currentPlayer == BLACK ? "Black" : "White") << "'s turn. ai move: " << row << " " << col << endl;


        if (board->PlacePiece(row, col, currentPlayer)) {
            
            pTemp = torch::zeros({1, BOARD_SIZE, BOARD_SIZE});
            for (auto child : ai->root->children){
                pTemp[0][child->lastMove.first][child->lastMove.second] = (child->visitCount.load() / ai->root->visitCount.load());
            }
            if (pTensor.size(0) == 0)
            {
                pTensor = pTemp.clone();
            }
            else {
                pTensor = torch::cat({ pTensor, pTemp });
            }
            

            result = board->IsGameOver(row, col);
            ai->Update(bestMove);
            /*cout << "ai updated" << endl;*/
            if (result != NOT_OVER) {
                board->PrintBoard();
                if (result == BLACK_WIN) {
                    cout << "Black wins!" << endl;
                }
                else if (result == WHITE_WIN) {
                    cout << "White wins!" << endl;
                }
                else {
                    cout << "It's a draw!" << endl;
                }
                gameOver = true;
            }
            else {
                board->SwitchPlayer();
                history.push_back(currentPlayer);
                if (currentPlayer == BLACK) {
                    boardTensor = torch::cat({ boardTensor, torch::cat({ torch::zeros({1, BOARD_SIZE, BOARD_SIZE}),board->DumpBoard() }).unsqueeze(0) });
                }
                else if (currentPlayer == WHITE) {
                    boardTensor = torch::cat({ boardTensor, torch::cat({ torch::ones({1, BOARD_SIZE, BOARD_SIZE}),board->DumpBoard(true) }).unsqueeze(0) });
                }
                
            }
        }
        else {
            cout << "Invalid move. Try again." << endl;
        }
    }
    // 计算价值
    vTensor = torch::zeros({ boardTensor.size(0), 1 });
    if (result != DRAW) {
        for (size_t i = 0; i < history.size(); i++)
        {
            if (result == BLACK_WIN)
            {
                if (history[i] == BLACK)
                {
                    vTensor[i][0] = -1;
                }
                else {
                    vTensor[i][0] = 1;
                }
            }
            else if (result == WHITE_WIN)
            {
                if (history[i] == BLACK)
                {
                    vTensor[i][0] = 1;
                }
                else {
                    vTensor[i][0] = -1;
                }
            }

        }
    }

    return make_pair(boardTensor, make_pair(pTensor, vTensor));
}