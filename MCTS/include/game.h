#pragma once
#include "gomoku.h"
#include "mcts.h"

// 游戏管理类
class ChessGame {
    private:
        
        Color currentPlayer;
        Color aiColor;
        
    public:
        GomokuBoard *board;
        MCTSAI ai;
        ChessGame(GomokuBoard *board, Color aiColor);
        ~ChessGame();
    
        void Start();
    
    private:
        vector<pair<int, int>> ParseInput(const string& input);
    };