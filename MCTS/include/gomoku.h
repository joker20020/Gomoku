#pragma once
#include <iostream>
#include <iomanip>
#include <vector>
#include <string>

#define BOARD_SIZE 15

using namespace std;

// 棋子颜色枚举
enum Color {
    EMPTY,  // 空
    BLACK,  // 黑棋
    WHITE   // 白棋
};

// 游戏结果枚举
enum GameResult {
    BLACK_WIN,  // 黑棋获胜
    WHITE_WIN,  // 白棋获胜
    DRAW,       // 平局
    NOT_OVER    // 游戏未结束
};

// 棋盘类
class GomokuBoard {
private:
    vector<vector<Color>> board;  // 15x15棋盘
    Color currentPlayer;         // 当前玩家

public:
    GomokuBoard();
    ~GomokuBoard();
    void InitializeBoard();      // 初始化棋盘
    void PrintBoard() const;     // 打印棋盘
    bool PlacePiece(int row, int col, Color color);  // 落子
    Color GetPiece(int row, int col) const;
    bool IsValidMove(int row, int col) const;        // 判断落子是否合法
    Color GetCurrentPlayer() const;  // 获取当前玩家
    void SwitchPlayer();         // 切换玩家

    // 判断游戏是否结束（仅检测当前落子位置）
    GameResult IsGameOver(int row, int col) const;

    // 评估棋盘状态
    double EvaluateBoard(GameResult result, Color currentPlayer) const;

    static vector<pair<int, int>> GenerateLegalMoves(const GomokuBoard& board, Color player);

private:
    // 判断当前落子是否形成五子连珠
    bool CheckWin(int row, int col, Color color) const;
};