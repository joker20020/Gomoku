#pragma once
#include <iostream>
#include <iomanip>
#include <vector>
#include <string>

#define BOARD_SIZE 15

using namespace std;

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
private:
    vector<vector<Color>> board;  // 15x15����
    Color currentPlayer;         // ��ǰ���

public:
    GomokuBoard();
    ~GomokuBoard();
    void InitializeBoard();      // ��ʼ������
    void PrintBoard() const;     // ��ӡ����
    bool PlacePiece(int row, int col, Color color);  // ����
    Color GetPiece(int row, int col) const;
    bool IsValidMove(int row, int col) const;        // �ж������Ƿ�Ϸ�
    Color GetCurrentPlayer() const;  // ��ȡ��ǰ���
    void SwitchPlayer();         // �л����

    // �ж���Ϸ�Ƿ����������⵱ǰ����λ�ã�
    GameResult IsGameOver(int row, int col) const;

    // ��������״̬
    double EvaluateBoard(GameResult result, Color currentPlayer) const;

    static vector<pair<int, int>> GenerateLegalMoves(const GomokuBoard& board, Color player);

private:
    // �жϵ�ǰ�����Ƿ��γ���������
    bool CheckWin(int row, int col, Color color) const;
};