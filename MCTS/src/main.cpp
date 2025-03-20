#include <ctime>
#include <cstdlib>
#include <torch/torch.h>
#include "mcts.h"
#include "game.h"
#include "model.h"
#include "train.h"
#include "gomoku.h"


int main() {

    torch::Device device = torch::Device(torch::kCPU);
    if (torch::cuda::is_available()) device = torch::kCUDA;
    else device = torch::kCPU;

    //GomokuBoard board = GomokuBoard();
    /*RlGomokuBoard board = RlGomokuBoard();*/
    shared_ptr<MCTSModel> model = make_shared<MCTSModel>();
    model->to(device);
    Trainer trainer = Trainer(model);
    trainer.Train();
    /*RlChessGame game = RlChessGame(&board, BLACK, model);
    
    game.Start();*/

    //cout << board.DumpBoard();

    /*MCTSModel model = MCTSModel();
    
    torch::Tensor x = torch::ones({1, 17 ,BOARD_SIZE ,BOARD_SIZE});
    cout << x << endl;
    cout << model.forward(x) << endl;*/

    /*torch::Tensor tensor = torch::rand({ 2, 3 }).cuda();
    std::cout << tensor << std::endl;
    cout << torch::cuda::is_available() << endl;*/

    return 0;
}