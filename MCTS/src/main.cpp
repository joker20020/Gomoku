#include <ctime>
#include <cstdlib>
#include <torch/torch.h>
#include <unistd.h>

#include "mcts.h"
#include "game.h"
#include "model.h"
#include "train.h"
#include "gomoku.h"


enum ArgType{
    VALUE_REQUIRED,
    VALUE_OPTIONAL, 
    VALUE_NONE   
};

struct Args {
    char shortcut;
    string longname;
    ArgType type;
    bool has;
    string value;
    string defaultValue;
};

void ParseArgs(int argc, char *argv[], Args (&args)[], int argCount);

int main(int argc, char *argv[]) {

    Args args[] = {
        {'m', "model", VALUE_OPTIONAL, false, "", "model/model0.pt"},
        {'d', "directory", VALUE_OPTIONAL, false, "", "model"},
        {'n', "num", VALUE_REQUIRED, false, "", "8"},
        {'t', "test", VALUE_NONE, false, "", ""}
    };

    ParseArgs(argc, argv, args, 4);
    cout << args[0].longname << ": value=" << args[0].value  << " has="  << args[0].has << endl;
    cout << args[1].longname << ": value=" << args[1].value  << " has="  << args[1].has << endl;
    cout << args[2].longname << ": value=" << args[2].value  << " has="  << args[2].has << endl;
    cout << args[3].longname << ": value=" << args[3].value  << " has="  << args[3].has << endl;

    torch::Device device = torch::Device(torch::kCPU);
    if (torch::cuda::is_available()) device = torch::kCUDA;
    else device = torch::kCPU;

    cout << device << endl;
    // cout << argv[1] << endl;
    

    //GomokuBoard board = GomokuBoard();
    /*RlGomokuBoard board = RlGomokuBoard();*/
    /*shared_ptr<MCTSModel> model1 = make_shared<MCTSModel>();
    shared_ptr<MCTSModel> model2 = make_shared<MCTSModel>();*/

    auto pool = make_shared<MCTSModelPool>(8);
    pool->Load("model/model0.pt");
    pool->to(device);

    Trainer trainer = Trainer(pool);
    trainer.Train();
    
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
    
    /*RlChessGame game = RlChessGame(&board, BLACK, model);
    
    game.Start();*/

    return 0;
}

void ParseArgs(int argc, char *argv[], Args (&args)[], int argCount){
    for(size_t i = 0; i < argCount; i++){
        for (size_t j = 0; j < argc; i++)
        {
            if(argv[j][0] == '-'){
                if(argv[j][1] == args[i].shortcut || argv[j] == ("-" + args[i].longname)){
                    switch (args[i].type)
                    {
                    case VALUE_NONE:
                        args[i].has = true;
                        break;
                    case VALUE_REQUIRED:
                        if(argv[j+1][0] != '-'){
                            args[i].value = argv[j+1];
                            args[i].has = true;
                        }
                        else{
                            args[i].value = "";
                            args[i].has = false;
                        }
                        break;
                    case VALUE_OPTIONAL:
                        args[i].has = true;
                        if(argv[j+1][0] != '-'){
                            args[i].value = argv[j+1];
                        }
                        else{
                            args[i].value = args[i].defaultValue;
                        }
                        break;
                    default:
                        break;
                    }
                }
            }
        }
        
    }
}
