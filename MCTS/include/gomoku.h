#pragma once
#include <iostream>
#include <iomanip>
#include <vector>
#include <string>
#include <torch/torch.h>

#define BOARD_SIZE 9
#define LAST_NUM 8

using namespace std;
using namespace torch::indexing;

// ������ɫö��
enum Color {
    EMPTY,  // ��
    BLACK,  // ����
    WHITE   // ����
};

// ��Ϸ���ö��
enum GameResult {
    BLACK_WIN,  // �����ʤ
    WHITE_WIN,  // �����ʤ
    DRAW,       // ƽ��
    NOT_OVER    // ��Ϸδ����
};

// ������
class GomokuBoard {
protected:
    vector<vector<Color>> board;  // 15x15����
    Color currentPlayer;         // ��ǰ���

public:
    GomokuBoard();
    ~GomokuBoard();
    void InitializeBoard();      // ��ʼ������
    void PrintBoard() const;     // ��ӡ����
    virtual bool PlacePiece(int row, int col, Color color);  // ����
    Color GetPiece(int row, int col) const;
    bool IsValidMove(int row, int col) const;        // �ж������Ƿ�Ϸ�
    Color GetCurrentPlayer() const;  // ��ȡ��ǰ���
    void SwitchPlayer();         // �л����

    // �ж���Ϸ�Ƿ����������⵱ǰ����λ�ã�
    GameResult IsGameOver(int row, int col) const;

    // ��������״̬
    double EvaluateBoard(GameResult result, Color currentPlayer) const;

    static vector<pair<int, int>> GenerateLegalMoves(const GomokuBoard& board, Color player);

    // �жϵ�ǰ�����Ƿ��γ���������
    bool CheckWin(int row, int col, Color color) const;

};

class RlGomokuBoard :public GomokuBoard {
private:
    vector<pair<pair<int, int>, Color>> lastMoves;

public:
    RlGomokuBoard();
    bool PlacePiece(int row, int col, Color color) override;  // ����
    torch::Tensor DumpBoard(bool swap = false);
};