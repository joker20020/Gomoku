#include <ctime>
#include <cstdlib>
#include <torch/torch.h>
#include "mcts.h"
#include "game.h"
#include "model.h"
#include "train.h"
#include "gomoku.h"


int main(int argc, char *argv[]) {

    torch::Device device = torch::Device(torch::kCPU);
    if (torch::cuda::is_available()) device = torch::kCUDA;
    else device = torch::kCPU;

    cout << device << endl;

    //GomokuBoard board = GomokuBoard();
    /*RlGomokuBoard board = RlGomokuBoard();*/
    /*shared_ptr<MCTSModel> model1 = make_shared<MCTSModel>();
    shared_ptr<MCTSModel> model2 = make_shared<MCTSModel>();*/

    auto pool = make_shared<MCTSModelPool>(8);
    pool->Load("net.pt");
    pool->to(device);
   
    //load model
    // torch::serialize::InputArchive archive;
    // archive.load_from("net.pt");
    // model->load(archive);
    
    //model->to(device);
    // torch::Tensor x = torch::ones({1, 17 ,BOARD_SIZE ,BOARD_SIZE});
    // for (size_t i = 0; i < 10; i++)
    // {
    //     std::chrono::system_clock::time_point start = std::chrono::system_clock::now();
    //     pair<torch::Tensor, torch::Tensor> nodeEvaluation = model->forward(x.to(device));
    //     std::chrono::system_clock::time_point end = std::chrono::system_clock::now();
    //     // 计算日期差
    //     std::chrono::duration<double> diff = end - start;
    //     // 输出日期差
    //     std::cout << "one forward is: " << diff.count() << " seconds" << std::endl;
    //     cout << nodeEvaluation.first.sizes() << nodeEvaluation.second.sizes() << endl;
    // }
    
    
    Trainer trainer = Trainer(pool);
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