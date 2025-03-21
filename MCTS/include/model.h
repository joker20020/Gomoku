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

enum ModelStatus {
	MODEL_FREE,
	MODEL_BUSY
};

class MCTSModelPool {
public:
	MCTSModelPool(int modelNum = 1, int resNum = 19, int pNum = 2, int vNum = 3);
	~MCTSModelPool();

    vector<shared_ptr<MCTSModel>> models;
	vector<ModelStatus> modelStatus;
	mutex mtx;
	

	pair<int, shared_ptr<MCTSModel>> GetModel(int modelIndex = -1);

	void to(torch::Device device);
	void Load(string path);
	void Sync(int modelIndex);
	void ReleaseModel(int modelIndex);
};
