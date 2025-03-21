#pragma once
#include <iostream>
#include <vector>
#include <map>
#include <cmath>
#include <ctime>
#include <cstdlib>
#include <algorithm>
#include <mutex>
#include <atomic>
#include <thread>
#include <random>

#include "model.h"
#include "gomoku.h"

using namespace std;


// MCTS 节点定义
class MCTSNode {
public:
    GomokuBoard board; // 当前棋盘状态
    Color currentPlayer; // 当前玩家
    MCTSNode* parent; // 父节点
    vector<MCTSNode*> children; // 子节点
    atomic<int> visitCount; // 访问次数
    atomic<double> totalScore; // 总得分
    atomic<double> virtualLoss; // 虚拟损失
    recursive_mutex mtx;
    pair<int, int> lastMove; // 记录最后移动

    MCTSNode();
    MCTSNode(const GomokuBoard& board, Color currentPlayer, MCTSNode* parent = nullptr);

    ~MCTSNode();

    // 判断是否为叶子节点
    virtual bool IsLeaf() const;

    // 判断是否为根节点
    virtual bool IsRoot() const;

    // 计算 UCB1 值
    virtual double UCB1(double explorationWeight = 1.414) const;

    // 选择最佳子节点
    virtual MCTSNode* SelectBestChild();

    // 扩展子节点
    virtual void Expand();

    // 随机模拟游戏
    virtual double Simulate();

    // 回溯更新节点
    virtual void Backpropagate(double score);

    // 获取最后移动
    virtual pair<int, int> GetLastMove() const;

    // 判断游戏是否结束
    virtual GameResult IsGameOver(const GomokuBoard& board, int row, int col);

    // 评估棋盘状态
    virtual double EvaluateBoard(GameResult result, Color player);

    // 打印节点对应棋盘
    virtual void Print();

protected:
    // 生成合法移动
    virtual vector<pair<int, int >> GenerateLegalMoves(const GomokuBoard& board, Color player);
    
};

// MCTS AI
class MCTSAI {
public:
    MCTSNode* root;

    MCTSAI();
    MCTSAI(const GomokuBoard& board, Color player);
    /* MCTSAI(const MCTSAI& other);
     MCTSAI& operator=(const MCTSAI& other);*/

    ~MCTSAI();
    // 运行 MCTS
    virtual void Run(int iterations);
    void ParallelRun(int iterations, int threadNum = 10);


    // 选择最佳移动
    virtual pair<int, int> GetBestMove(bool random = false);

    // 自动更新节点
    virtual void AutoUpdate();

    // 手动更新节点
    virtual void Update(pair<int, int> move);

    // 选择节点
    MCTSNode* Select(MCTSNode* node);

};

class RlMCTSNode :public MCTSNode {
protected:
    RlGomokuBoard board;
    shared_ptr<MCTSModelPool> modelPool;
    atomic<double> p;
    atomic<double> c;
    
public:
    vector<RlMCTSNode*> children; // 子节点

    RlMCTSNode();
    RlMCTSNode(const RlGomokuBoard& board, Color currentPlayer, shared_ptr<MCTSModelPool> modelPool, double p = 0.0, RlMCTSNode* parent = nullptr, double c = 3);
    ~RlMCTSNode();

    // 扩展子节点
    void Expand() override;
    double UCB1(double explorationWeight = 1.414) const override;
    RlMCTSNode* SelectBestChild() override;
    bool IsLeaf() const override;
    
};

class RlMCTSAI :public MCTSAI{

public:
    RlMCTSNode* root;
    shared_ptr<MCTSModelPool> model;

    RlMCTSAI();
    RlMCTSAI(const RlGomokuBoard& board, Color player, shared_ptr<MCTSModelPool> modelPool);
    ~RlMCTSAI();
    // 运行 MCTS
    void Run(int iterations) override;
    pair<int, int> GetBestMove(bool random = false) override;
    void Update(pair<int, int> move) override;
    
};