#include "mcts.h"

MCTSNode::MCTSNode() {
    this->currentPlayer = BLACK;
    this->parent = nullptr;
    this->visitCount.store(0);
    this->totalScore.store(0);
    this->virtualLoss.store(0);
    this->lastMove = pair<int, int>{ -1,-1 };
    this->children = {};
}

MCTSNode::MCTSNode(const GomokuBoard& board, Color currentPlayer, MCTSNode* parent){
    this->board = board;
    this->currentPlayer = currentPlayer;
    this->parent = parent;
    this->visitCount.store(0);
    this->totalScore.store(0);
    this->virtualLoss.store(0);
    this->lastMove = pair<int, int>{ -1,-1 };
}

MCTSNode::~MCTSNode() {
    for (auto child : children) {
        delete child;
    }
}

// 判断是否为叶子节点
bool MCTSNode::IsLeaf() const {
    return children.empty();
}

// 判断是否为根节点
bool MCTSNode::IsRoot() const {
    return parent == nullptr;
}

// 计算 UCB1 值
double MCTSNode::UCB1(double explorationWeight) const{
    if (visitCount == 0) return numeric_limits<double>::max();
    return (totalScore.load() / visitCount.load()) + explorationWeight * sqrt(log(parent->visitCount.load()) / visitCount.load()) + virtualLoss.load();
}

// 选择最佳子节点
MCTSNode* MCTSNode::SelectBestChild() {
    lock_guard<recursive_mutex> lock(mtx);
    MCTSNode* bestChild = *max_element(children.begin(), children.end(), [](MCTSNode* a, MCTSNode* b) {
        return a->UCB1() < b->UCB1();
    });
    bestChild->virtualLoss.store(bestChild->virtualLoss.load() - 1);
    return bestChild;
}

// 扩展子节点
void MCTSNode::Expand() {
    lock_guard<recursive_mutex> lock(mtx);
    if (!IsLeaf()) {
        MCTSNode *node = this->SelectBestChild();
        while (!node->IsLeaf())
        {
            node = node->SelectBestChild();
        }
        node->Expand();
        return;
    }
    vector<pair<int, int>> moves = GenerateLegalMoves(board, currentPlayer);
    for (const auto& move : moves) {
        GomokuBoard newBoard = board;
        newBoard.PlacePiece(move.first, move.second, currentPlayer);
        newBoard.SwitchPlayer();
        MCTSNode* child = new MCTSNode(newBoard, (currentPlayer == WHITE) ? BLACK : WHITE, this);
        child->lastMove = move; // 记录移动
        children.push_back(child);
    }
}

// 随机模拟游戏
double MCTSNode::Simulate() {
    GomokuBoard simBoard = board;
    Color simPlayer = currentPlayer;
    GameResult result = IsGameOver(simBoard, lastMove.first, lastMove.second);
    while (result == NOT_OVER) {
        vector<pair<int, int>> moves = GenerateLegalMoves(simBoard, simPlayer);
        if (moves.empty()) break;
        auto randomMove = moves[rand() % moves.size()];
        simBoard.PlacePiece(randomMove.first, randomMove.second, simPlayer);
        // cout << "move:" << randomMove.first.first << "," << randomMove.first.second << "->" << randomMove.second.first << "," << randomMove.second.second << endl;
        // cout << "noEatCount:" << noEatCount << endl;
        simPlayer = (simPlayer == WHITE) ? BLACK : WHITE;
        simBoard.SwitchPlayer();
        // simBoard.Print();
        result = IsGameOver(simBoard, randomMove.first, randomMove.second);
    }
    return EvaluateBoard(result, currentPlayer);
}

// 回溯更新节点
void MCTSNode::Backpropagate(double score) {
    visitCount.fetch_add(1);
    totalScore.store(totalScore + score);
    virtualLoss.store(virtualLoss + 1);
    if (!IsRoot()) parent->Backpropagate(-score);
}

// 生成合法移动
vector<pair<int, int>> MCTSNode::GenerateLegalMoves(const GomokuBoard& board, Color player) {
    return board.GenerateLegalMoves(board, player);
}

// 判断游戏是否结束
GameResult MCTSNode::IsGameOver(const GomokuBoard& board, int row, int col) {
    if (row == -1 && col == -1) return NOT_OVER;
    return board.IsGameOver(row, col);
}

// 评估棋盘状态
double MCTSNode::EvaluateBoard(GameResult result, Color currentPlayer) {
    return board.EvaluateBoard(result, currentPlayer);
    
}

// 打印节点对应棋盘
void MCTSNode::Print(){
    board.PrintBoard();
}

// 获取上一次移动
pair<int, int> MCTSNode::GetLastMove() const {
    return lastMove;
}

MCTSAI::MCTSAI() {
    root = new MCTSNode();
}

MCTSAI::MCTSAI(const GomokuBoard &board, Color player) {
    root = new MCTSNode(board, player);
}

//MCTSAI::MCTSAI(const MCTSAI &other) {
//    root = other.root;
//}
//MCTSAI& MCTSAI::operator=(const MCTSAI& other){
//    root = other.root;
//    return *this;
//}
MCTSAI::~MCTSAI() {
    delete root;
}

// 运行 MCTS
void MCTSAI::Run(int iterations) {
    for (int i = 0; i < iterations; ++i) {
         //cout << "Iteration: " << i + 1 << "/"  << iterations << '\r';
        MCTSNode* node = Select(root);
        if (!node->IsLeaf()) {
            node = node->SelectBestChild();
        }
        if (node->IsGameOver(node->board, node->lastMove.first, node->lastMove.second) == NOT_OVER && node->IsLeaf()) {
            node->Expand();
            node = node->children[rand() % node->children.size()];
            node->virtualLoss.store(node->virtualLoss.load() - 1);
        }
        double score = node->Simulate();
        node->Backpropagate(score);
    }
}

// 多线程运行 MCTS
void MCTSAI::ParallelRun(int iterations, int threadNum) {
    vector<thread> threads;
    int threadIterations = iterations / threadNum;
    srand(time(0));
    for (int i = 0; i < threadNum; i++)
    {
        threads.push_back(
            thread(&MCTSAI::Run, this, threadIterations)
        );
    }
    for(auto &thread : threads){
        thread.join();
    }
    
}

// 选择最佳移动
pair<int, int> MCTSAI::GetBestMove(bool random) {
    MCTSNode* bestChild = *max_element(root->children.begin(), root->children.end(), [](MCTSNode* a, MCTSNode* b) {
        return a->visitCount < b->visitCount;
    });
    return bestChild->GetLastMove();
}

// 选择节点
MCTSNode* MCTSAI::Select(MCTSNode* node) {
    while (!node->IsLeaf()) {
        node = node->SelectBestChild();
    }
    return node;
}

// 更新节点
void MCTSAI::AutoUpdate() {
    if (root->children.size() == 0){
        Run(1);
    }
    vector<MCTSNode*>::iterator bestChildIterator = max_element(root->children.begin(), root->children.end(), [](MCTSNode* a, MCTSNode* b) {
        return a->visitCount < b->visitCount;
    });
    MCTSNode* bestChild = *bestChildIterator;
    int bestChildIndex = bestChildIterator - root->children.begin();

    root->children.erase(root->children.begin() + bestChildIndex);

    MCTSNode* temp = this->root;
    root = bestChild;
    root->parent = nullptr;
    thread* delThread = new thread([temp] {
        delete temp;
        });
    //delete root;
}

void MCTSAI::Update(pair<int, int> move) {
    size_t i = 0;
    if (root->children.size() == 0){
        Run(1);
    }
    for(;i < root->children.size(); ++i){
        if(root->children[i]->GetLastMove() == move){
            break;
        }
    }
    MCTSNode* bestChild = root->children[i];

    root->children.erase(root->children.begin() + i);
    
    MCTSNode* temp = this->root;
    root = bestChild;
    root->parent = nullptr;
    thread* delThread = new thread([temp] {
        delete temp;
        });
    //delThread->detach();
    //delete root;
    
}

RlMCTSNode::RlMCTSNode() :MCTSNode() {
}


RlMCTSNode::RlMCTSNode(const RlGomokuBoard & board, Color currentPlayer, shared_ptr<MCTSModelPool> modelPool, double p, RlMCTSNode* parent, double c):MCTSNode(board, currentPlayer, parent) {
    this->board = board;
    this->modelPool = modelPool;
    this->p = p;
    this->c = c;
}

RlMCTSNode::~RlMCTSNode() {
    for (auto child : children) {
        delete child;
    }
}

void RlMCTSNode::Expand() {
    lock_guard<recursive_mutex> lock(mtx);
    torch::Tensor evaluateBoard;
    if (!IsLeaf()) {
        MCTSNode* node = this->SelectBestChild();
        while (!node->IsLeaf())
        {
            node = node->SelectBestChild();
        }
        node->Expand();
        return;
    }
    // 模型评估局面，input[color, n, n-1, ... ,n-7] return (p,v)
    if (currentPlayer == BLACK) {
        evaluateBoard = torch::cat({ torch::zeros({1, BOARD_SIZE, BOARD_SIZE}),board.DumpBoard() }).unsqueeze(0);
    }
    else if (currentPlayer == WHITE) {
        evaluateBoard = torch::cat({ torch::ones({1, BOARD_SIZE, BOARD_SIZE}),board.DumpBoard(true) }).unsqueeze(0);
    }
    auto modelPair = modelPool->GetModel();
    auto modelIndex = modelPair.first;
    auto model = modelPair.second;
    auto device = model->parameters()[0].device();
    // std::chrono::system_clock::time_point start = std::chrono::system_clock::now();
    pair<torch::Tensor, torch::Tensor> nodeEvaluation = model->forward(evaluateBoard.to(device));
    modelPool->ReleaseModel(modelIndex);
    // std::chrono::system_clock::time_point end = std::chrono::system_clock::now();
    // // 计算日期差
    // std::chrono::duration<double> diff = end - start;
    // // 输出日期差
    // std::cout << "one forward is: " << diff.count() << " seconds" << std::endl;
    // cout << evaluateBoard.sizes() << endl;
    
    // 取tensor[0]
    nodeEvaluation.first = model->softMax(nodeEvaluation.first).squeeze();
    nodeEvaluation.second = nodeEvaluation.second.squeeze();

    vector<pair<int, int>> moves = GenerateLegalMoves(board, currentPlayer);
    for (const auto& move : moves) {
        RlGomokuBoard newBoard = board;
        newBoard.PlacePiece(move.first, move.second, currentPlayer);
        newBoard.SwitchPlayer();
        RlMCTSNode* child = new RlMCTSNode(
            newBoard, 
            (currentPlayer == WHITE) ? BLACK : WHITE, 
            modelPool, 
            nodeEvaluation.first[move.first * BOARD_SIZE + move.second].item<double>(),
            this);
        child->lastMove = move; // 记录移动
        children.push_back(child);
    }
    
    Backpropagate(nodeEvaluation.second.item<double>());
}

double RlMCTSNode::UCB1(double explorationWeight) const{
    // U + Q
    if(visitCount.load() == 0) return c.load() * p.load() * sqrt(parent->visitCount.load()) / (1 + visitCount.load()) + virtualLoss.load();
    return c.load() * p.load() * sqrt(parent->visitCount.load()) / (1 + visitCount.load()) + (totalScore.load() / visitCount.load()) + virtualLoss.load();
}

RlMCTSNode* RlMCTSNode::SelectBestChild() {
    lock_guard<recursive_mutex> lock(mtx);
    RlMCTSNode* bestChild = *max_element(children.begin(), children.end(), [](MCTSNode* a, MCTSNode* b) {
        return a->UCB1() < b->UCB1();
        });
    bestChild->virtualLoss.store(bestChild->virtualLoss.load() - 1);
    return bestChild;
}

bool RlMCTSNode::IsLeaf() const {
    return children.empty();
}

RlMCTSAI::RlMCTSAI() {
}

RlMCTSAI::RlMCTSAI(const RlGomokuBoard& board, Color player, shared_ptr<MCTSModelPool> modelPool) {
    model = modelPool;
    root = new RlMCTSNode(board, player, model);
}

RlMCTSAI::~RlMCTSAI() {
    delete root;
}

// 运行 MCTS
void RlMCTSAI::Run(int iterations) {
    GameResult result;
    for (int i = 0; i < iterations; ++i) {
        // std::chrono::system_clock::time_point start = std::chrono::system_clock::now();
        //cout << "Iteration: " << i + 1 << "/"  << iterations << '\r';
        MCTSNode* node = Select(root);
        result = node->IsGameOver(node->board, node->lastMove.first, node->lastMove.second);
        if (result == NOT_OVER) {
            node->Expand();
        }
        else if(result == DRAW){
            node->Backpropagate(0);
        }
        else{
            node->Backpropagate(1);
        }
        // std::chrono::system_clock::time_point end = std::chrono::system_clock::now();
        // // 计算日期差
        // std::chrono::duration<double> diff = end - start;

        // // 输出日期差
        // std::cout << "Date difference is: " << diff.count() << " seconds" << std::endl;
    }
}

pair<int, int> RlMCTSAI::GetBestMove(bool random) {
    if (random) {
        std::vector<double> weights = {};
        for(auto child : root->children){
            weights.push_back(child->visitCount.load());
        }

        // 创建随机数生成器
        std::random_device rd;
        std::mt19937 gen(rd());

        // 创建带权重的分布
        std::discrete_distribution<> dist(weights.begin(), weights.end());

        int random_number = dist(gen);
        
        return root->children[random_number]->GetLastMove();
    }
    RlMCTSNode* bestChild = *max_element(root->children.begin(), root->children.end(), [](RlMCTSNode* a, RlMCTSNode* b) {
        return a->visitCount.load() < b->visitCount.load();
        });
    return bestChild->GetLastMove();
}

void RlMCTSAI::Update(pair<int, int> move) {
    size_t i = 0;
    if (root->children.size() == 0) {
        Run(1);
    }
    for (; i < root->children.size(); ++i) {
        if (root->children[i]->GetLastMove() == move) {
            break;
        }
    }
    RlMCTSNode* bestChild = root->children[i];

    root->children.erase(root->children.begin() + i);

    RlMCTSNode* temp = this->root;
    root = bestChild;
    root->parent = nullptr;
    thread* delThread = new thread([temp] {
        delete temp;
        });

}
