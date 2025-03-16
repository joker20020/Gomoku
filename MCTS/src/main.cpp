#include <ctime>
#include <cstdlib>
//#include <torch/torch.h>
#include "mcts.h"
#include "game.h"
#include "gomoku.h"


int main() {
    GomokuBoard board = GomokuBoard();
    ChessGame game = ChessGame(&board, BLACK);
    game.Start();

    /*torch::Tensor tensor = torch::rand({ 2, 3 }).cuda();
    std::cout << tensor << std::endl;
    cout << torch::cuda::is_available() << endl;*/

    return 0;
}