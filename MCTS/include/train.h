#pragma once
#include <torch/torch.h>
#include "model.h"
#include "gomoku.h"
#include "mcts.h"
#include "game.h"

using namespace torch::indexing;
class GomokuDataset : public torch::data::Dataset<GomokuDataset, torch::data::Example<torch::Tensor, pair<torch::Tensor, torch::Tensor>>> {
public:
	torch::Tensor board;
	torch::Tensor p;
	torch::Tensor v;
	GomokuDataset(torch::Tensor board, torch::Tensor p, torch::Tensor v);

	torch::data::Example<torch::Tensor, pair<torch::Tensor, torch::Tensor>> get(size_t index) override;

    torch::optional<size_t> size() const override;
};

class Trainer {
public:
    shared_ptr<MCTSModelPool> modelPool;

	int selfPlayTimes;
	int epoch;
	int batchSize;
	double lr;

	Trainer(shared_ptr<MCTSModelPool> modelPool, int selfPlayTimes = 10000, int epoch = 25, int batchSize = 64, double lr = 0.01);

    void Train();
};