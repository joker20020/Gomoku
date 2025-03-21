#include "train.h"

GomokuDataset::GomokuDataset(torch::Tensor board, torch::Tensor p, torch::Tensor v) {
	this->board = board;;
    this->p = p;
    this->v = v;
}

torch::data::Example<torch::Tensor, pair<torch::Tensor, torch::Tensor>> GomokuDataset::get(size_t index) {
    return {board[index], {p[index], v[index]}};
}

torch::optional<size_t> GomokuDataset::size() const {
    return board.size(0);
}


Trainer::Trainer(shared_ptr<MCTSModelPool> modelPool, int selfPlayTimes, int epoch, int batchSize, double lr) {
    this->modelPool = modelPool;
    this->epoch = epoch;
    this->batchSize = batchSize;
    this->lr = lr;
    this->selfPlayTimes = selfPlayTimes;
}

void Trainer::Train() {
    int memorySize = 10;
    int dataSize = 20;
    torch::Tensor boardTensors, pTensors, vTensors;
    boardTensors = torch::tensor({});
    pTensors = torch::tensor({});
    vTensors = torch::tensor({});

    for (size_t i = 0; i < selfPlayTimes; i++)
    {
        while (boardTensors.size(0) < dataSize)
        {
            shared_ptr<RlGomokuBoard> board = make_shared<RlGomokuBoard>();
            shared_ptr<RlChessGame> game = make_shared<RlChessGame>(board, BLACK, modelPool);
            pair<torch::Tensor, pair<torch::Tensor, torch::Tensor>> trainResult = game->TrainStart();
            boardTensors = torch::cat({ boardTensors, trainResult.first}, 0);
            pTensors = torch::cat({pTensors, trainResult.second.first}, 0);
            vTensors = torch::cat({vTensors, trainResult.second.second}, 0);
            int64_t originCount = boardTensors.size(0);
            // data augmente
            for (size_t dataCount = originCount - memorySize; dataCount < originCount; dataCount++)
            {
                // rot 90
                boardTensors = torch::cat({ boardTensors, torch::rot90(boardTensors[dataCount], 1, {1, 2}).unsqueeze(0)}, 0);
                // cout << "board OK" << endl;
                pTensors = torch::cat({pTensors, torch::rot90(pTensors[dataCount], 1, {0, 1}).unsqueeze(0)}, 0);
                // cout << "p OK" << endl;
                vTensors = torch::cat({vTensors, vTensors[dataCount].unsqueeze(0)}, 0);
                // cout << "v OK" << endl;
                // rot 180
                boardTensors = torch::cat({ boardTensors, torch::rot90(boardTensors[dataCount], 2, {1, 2}).unsqueeze(0)}, 0);
                pTensors = torch::cat({pTensors, torch::rot90(pTensors[dataCount], 2, {0, 1}).unsqueeze(0)}, 0);
                vTensors = torch::cat({vTensors, vTensors[dataCount].unsqueeze(0)}, 0);
                // rot 270
                boardTensors = torch::cat({ boardTensors, torch::rot90(boardTensors[dataCount], 3, {1, 2}).unsqueeze(0)}, 0);
                pTensors = torch::cat({pTensors, torch::rot90(pTensors[dataCount], 3, {0, 1}).unsqueeze(0)}, 0);
                vTensors = torch::cat({vTensors, vTensors[dataCount].unsqueeze(0)}, 0);
                // flip lr
                boardTensors = torch::cat({ boardTensors, torch::flip(boardTensors[dataCount], {2}).unsqueeze(0)}, 0);
                pTensors = torch::cat({pTensors, torch::flip(pTensors[dataCount], {1}).unsqueeze(0)}, 0);
                vTensors = torch::cat({vTensors, vTensors[dataCount].unsqueeze(0)}, 0);
                // flip ud
                boardTensors = torch::cat({ boardTensors, torch::flip(boardTensors[dataCount], {1}).unsqueeze(0)}, 0);
                pTensors = torch::cat({pTensors, torch::flip(pTensors[dataCount], {0}).unsqueeze(0)}, 0);
                vTensors = torch::cat({vTensors, vTensors[dataCount].unsqueeze(0)}, 0);
            }
            
        }

        cout << "Total data size: " << boardTensors.size(0) << '\n';

        // ѵ��ѭ��
        auto modelPair = modelPool->GetModel();
        auto modelIndex = modelPair.first;
        auto model = modelPair.second;
        pTensors = pTensors.flatten(1);
        GomokuDataset dataset = GomokuDataset(boardTensors, pTensors, vTensors);
        auto data_loader = torch::data::make_data_loader<torch::data::samplers::RandomSampler>(std::move(dataset), batchSize);
        torch::optim::Adam optimizer(model->parameters(), torch::optim::AdamOptions({lr}).weight_decay(1e-4));
        auto device = model->parameters()[0].device();
        
        for (size_t j = 1; j <= epoch; ++j) {

            cout << "Trainning epoch " << j << '\r';
            
            // Iterate the data loader to yield batches from the dataset.
            for (auto& batch : *data_loader) {
                // Reset gradients.
                optimizer.zero_grad();
                torch::Tensor datas = torch::tensor({});
                pair<torch::Tensor, torch::Tensor> targets = { torch::tensor({}),  torch::tensor({}) };
                for (auto& data : batch) {
                    datas = torch::cat({datas, data.data.unsqueeze(0)}, 0);
                    targets.first = torch::cat({targets.first, data.target.first.unsqueeze(0) }, 0);
                    targets.second = torch::cat({targets.second, data.target.second.unsqueeze(0) }, 0);
                }
                // Execute the model on the input data.
                pair<torch::Tensor, torch::Tensor> prediction = model->forward(datas.to(device));
                // Compute a loss value to judge the prediction of our model.
                torch::Tensor creLoss = torch::nn::functional::cross_entropy(prediction.first, targets.first.to(device));
                torch::Tensor mseLoss = torch::nn::functional::mse_loss(prediction.second, targets.second.to(device));
                torch::Tensor totalLoss = creLoss + mseLoss;
                // Compute gradients of the loss w.r.t. the parameters of our model.
                totalLoss.backward();
                // Update the parameters based on the calculated gradients.
                optimizer.step();
                // Output the loss and checkpoint every 100 batches.
                if (++i % 2 == 0) {
                    std::cout << "Epoch: " << j << " | Self play: " << i
                        << " | Loss: " << totalLoss.item<float>() << std::endl;
                    // Serialize your model periodically as a checkpoint.
                    // std::stringstream modelName;
                    // modelName << "model/net";
                    // modelName << i;
                    // modelName << ".pt";
                    torch::save(model, "net.pt");
                }
            }
        }

        boardTensors = boardTensors.index({Slice(-memorySize, None)});
        pTensors = pTensors.index({Slice(-memorySize, None)});
        vTensors = vTensors.index({Slice(-memorySize, None)});

        modelPool->ReleaseModel(modelIndex);
        modelPool->Sync(modelIndex);
    }

}