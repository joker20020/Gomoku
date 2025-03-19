#pragma once

#include <torch/torch.h>
#include "gomoku.h"

using namespace torch::nn;


class ResBlock :public torch::nn::Module {
public:

	torch::nn::Sequential net1;
	torch::nn::Sequential net2;
	torch::nn::ReLU relu;

	ResBlock(int inChannel, int outChannel);
	torch::Tensor forward(torch::Tensor x);

};


class MCTSModel : public torch::nn::Module {
public:
	MCTSModel(int resNum = 19, int pNum = 2, int vNum = 3);
	pair<torch::Tensor, torch::Tensor> forward(torch::Tensor x);

	torch::nn::Sequential net;
	torch::nn::Sequential pHead;
	torch::nn::Sequential vHead;
	torch::nn::Softmax softMax;

	mutex mtx;
};
