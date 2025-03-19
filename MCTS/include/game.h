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
        ChessGame();
        ChessGame(GomokuBoard *board, Color aiColor);
        ~ChessGame();
    
        virtual void Start();
    
    private:
        vector<pair<int, int>> ParseInput(const string& input);
};

class RlChessGame: public ChessGame {
private:

    Color currentPlayer;
    Color aiColor;

public:
    RlGomokuBoard* board;
    RlMCTSAI ai;
    RlChessGame(RlGomokuBoard* board, Color aiColor, shared_ptr<MCTSModel> model);
    ~RlChessGame();

    void Start() override;

private:
    
};
