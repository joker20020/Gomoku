#pragma once
#include "model.h"
#include "gomoku.h"
#include "mcts.h"
#include <torch/torch.h>

class GomokuDataset : public torch::data::datasets::Dataset<GomokuDataset> {
public:
	GomokuDataset();
};