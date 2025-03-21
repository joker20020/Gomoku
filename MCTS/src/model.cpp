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
	for (size_t i = 0; i < vNum; i++)
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
	// lock_guard<mutex> lock(mtx);
	pair<torch::Tensor, torch::Tensor> result;
	torch::Tensor feature = net->forward(x);
	result.first = pHead->forward(feature);
	result.second = vHead->forward(feature);
	return result;
}

MCTSModelPool::MCTSModelPool(int modelNum, int resNum, int pNum, int vNum) {
	for (size_t i = 0; i < modelNum; i++)
	{
		models.push_back(make_shared<MCTSModel>(resNum, pNum, vNum));
		modelStatus.push_back(MODEL_FREE);
	}
}

MCTSModelPool::~MCTSModelPool() {
	for (size_t i = 0; i < models.size(); i++)
	{
		models[i].reset();
	}
}

pair<int, shared_ptr<MCTSModel>> MCTSModelPool::GetModel(int modelIndex) {
	std::unique_lock<mutex> lock(mtx, std::defer_lock);
	if (modelIndex != -1) {
		modelStatus[modelIndex] = MODEL_BUSY;
		return {modelIndex, models[modelIndex] };
	}
	while (true)
	{
		lock.lock();
		for (size_t i = 0; i < modelStatus.size(); i++) {
			if (modelStatus[i] == MODEL_FREE) {
				modelStatus[i] = MODEL_BUSY;
				return { i, models[i] };
			}
		}
		lock.unlock();
	}
}

void MCTSModelPool::to(torch::Device device) {
	for (size_t i = 0; i < models.size(); i++)
	{
		models[i]->to(device);
	}
}

void MCTSModelPool::Load(string path) {
	torch::serialize::InputArchive archive;
	archive.load_from(path);
	for (size_t i = 0; i < models.size(); i++)
	{
		models[i]->load(archive);
	}
}

void MCTSModelPool::Sync(int modelIndex) {
	std::lock_guard<mutex> lock(mtx);
	torch::save(models[modelIndex], "net.temp");
	/*cout << models[modelIndex]->parameters()[1] << endl;
	cout << models[1]->parameters()[1] <<endl;*/
	Load("net.temp");
	remove("net.temp");
	/*cout << models[modelIndex]->parameters()[1] << endl;
	cout << models[1]->parameters()[1] << endl;*/

	/*auto newParams = models[modelIndex]->named_parameters();
	for (size_t i = 0; i < models.size(); i++)
	{
		auto oldParams = models[i]->named_parameters();
		for (auto& val : oldParams) {
			auto name = val.key();
			auto* t = newParams.find(name);
			if (t != nullptr) {
				t->copy_(val.value());
			}
			else {
				t = oldParams.find(name);
				if (t != nullptr) {
					t->copy_(val.value());
				}
			}
		}
	}*/

	
}

void MCTSModelPool::ReleaseModel(int modelIndex) {
    modelStatus[modelIndex] = MODEL_FREE;
}
