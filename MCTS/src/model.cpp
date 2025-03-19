#include "model.h"

ResBlock::ResBlock(int inChannel, int outChannel): net1(
	Conv2d(Conv2dOptions(inChannel, outChannel, 3).stride(1).padding(1)),
	BatchNorm2d(outChannel),
	ReLU(),
	Conv2d(Conv2dOptions(outChannel, outChannel, 3).stride(1).padding(1)),
	BatchNorm2d(outChannel)
), net2(Conv2d(Conv2dOptions(inChannel, outChannel, 1))), relu() {
	register_module("net1", net1);
	register_module("net2", net2);
	register_module("relu", relu);
}



torch::Tensor ResBlock::forward(torch::Tensor x) {
	x = relu(net1->forward(x) + net2->forward(x));
	return x;
}

MCTSModel::MCTSModel(int resNum, int pNum, int vNum): net(), pHead(), vHead(), softMax(SoftmaxOptions(1)) {
	net->push_back(Conv2d(Conv2dOptions(LAST_NUM * 2 + 1, 256, 1)));
	net->push_back(BatchNorm2d(256));
	net->push_back(ReLU());
	for (size_t i = 0; i < resNum; i++)
	{
		net->push_back(ResBlock(256, 256));
	}

	pHead->push_back(Conv2d(Conv2dOptions(256, 4, 1)));
	pHead->push_back(BatchNorm2d(4));
	pHead->push_back(ReLU());
	for (size_t i = 0; i < pNum; i++)
	{
		pHead->push_back(Conv2d(Conv2dOptions(4, 4, 1)));
		pHead->push_back(BatchNorm2d(4));
		pHead->push_back(ReLU());
	}
	pHead->push_back(Flatten());
	pHead->push_back(Linear(BOARD_SIZE * BOARD_SIZE * 4, BOARD_SIZE * BOARD_SIZE));

	vHead->push_back(Conv2d(Conv2dOptions(256, 1, 1)));
	vHead->push_back(BatchNorm2d(1));
	vHead->push_back(ReLU());
	for (size_t i = 0; i < pNum; i++)
	{
		vHead->push_back(Conv2d(Conv2dOptions(1, 1, 1)));
		vHead->push_back(BatchNorm2d(1));
		vHead->push_back(ReLU());
	}
	vHead->push_back(Flatten());
	vHead->push_back(Linear(BOARD_SIZE * BOARD_SIZE, 256));
	vHead->push_back(ReLU());
	vHead->push_back(Linear(256, 1));
	vHead->push_back(Tanh());


	register_module("net", net);
	register_module("pHead", pHead);
	register_module("vHead", vHead);
	register_module("softMax", softMax);
}

// return (p, v)
pair<torch::Tensor, torch::Tensor> MCTSModel::forward(torch::Tensor x) {
	//lock_guard<mutex> lock(mtx);
	pair<torch::Tensor, torch::Tensor> result;
	torch::Tensor feature = net->forward(x);
	result.first = pHead->forward(feature);
	result.second = vHead->forward(feature);
	return result;
}