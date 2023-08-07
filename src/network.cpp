#include "network.hpp"

Network::Network()
{
}

Network::Network(std::vector<int> layerSize)
{
    // 初始化层数
    for (int i = 0; i < layerSize.size(); ++i)
    {
        // 如果是第一层，则初始化第一层
        if (i == 0)
        {
            layers.push_back(std::make_shared<Layer>(0, layerSize[i]));
        }
        // 如果是最后一层，则初始化最后一层
        else
        {
            layers.push_back(std::make_shared<Layer>(layerSize[i - 1], layerSize[i]));
        }
    }
}

// 计算损失函数
double Network::calculateLoss(std::vector<double> output)
{
    // 初始化损失值
    double loss = 0.0;
    // 遍历每一层的输出
    for (int i = 0; i < layers.back()->neurons.size(); ++i)
    {
        // 计算损失函数
        loss += pow((layers.back()->neurons.at(i)->output - output[i]), 2);
    }
    // 返回损失值
    return loss / (2 * output.size());
}

// 激活函数
double Network::Sigmoid(double x)
{
    // 返回激活函数
    return 1 / (1 + exp(-x));
}

// 激活函数的梯度
double Network::SigmoidDerivative(double y)
{
    // 返回激活函数的梯度
    return y * (1 - y);
}

// ReLu函数
double Network::ReLu(double x)
{
    // 返回ReLu函数
    return std::max(x, 0.0);
}

// ReLu函数的梯度
double Network::ReLuDerivative(double y)
{
    // 返回ReLu函数的梯度
    return y > 0 ? 1.0 : 0.0;
}

// LeakyReLu函数
double Network::LeakyReLu(double d)
{
    // 返回LeakyReLu函数
    return std::max(0.01 * d, d);
}

double Network::LeakyReLuDerivative(double d)
{
    // 返回LeakyReLu函数的梯度
    return d > 0 ? 1 : 0.01;
}

double Network::Tanh(double x)
{
    // double a = exp(x);
    // double b = exp(-x);
    // return (a - b) / (a + b);
    return tanh(x);
}

// 激活函数的梯度
double Network::TanhDerivative(double x)
{
    // double a = exp(x);
    // double b = exp(-x);
    // return 1 - pow((a - b) / (a + b), 2);
    return 1 - pow(tanh(x), 2);
}

double Network::activate(double x)
{
    return Sigmoid(x);
}

double Network::activateDerivative(double y)
{
    // 返回激活函数的梯度
    return SigmoidDerivative(y);
}

void Network::bprop(std::vector<double> output)
{
    // 遍历层数
    for (int i = layers.size() - 1; i >= 0; i--)
    {
        // 获取当前层
        auto l = layers.at(i);
        // 初始化错误
        std::vector<double> errors;
        // 如果不是最后一层
        if (i != layers.size() - 1)
        {
            // 遍历当前层的每一个神经元
            for (int j = 0; j < l->neurons.size(); j++)
            {
                // 初始化错误
                double error = 0;
                // 遍历当前层的下一层的每一个神经元
                for (auto n : layers.at(i + 1)->neurons)
                {
                    // 计算错误
                    error += n->weights.at(j) * n->delta;
                }
                // 将错误添加到错误列表
                errors.push_back(error);
            }
        }
        // 如果是最后一层
        else
        {
            // 遍历当前层的每一个神经元
            for (int j = 0; j < l->neurons.size(); j++)
            {
                // 计算错误
                errors.push_back(output.at(j) - l->neurons.at(j)->output);
            }
        }
        // 遍历当前层的每一个神经元
        for (int j = 0; j < l->neurons.size(); j++)
        {
            // 获取当前神经元
            auto n = l->neurons.at(j);
            // 计算梯度
            n->delta = errors.at(j) * activateDerivative(n->output);
        }
    }
}

std::vector<double> Network::collectGrad()
{
    // 初始化梯度向量
    std::vector<double> grad;
    // 初始化输入向量
    std::vector<double> input;
    // 遍历层
    for (int i = 1; i < layers.size(); ++i)
    {
        // 获取当前层
        auto l = layers.at(i);

        // 遍历上一层的节点
        for (auto n : layers.at(i - 1)->neurons)
        {
            // 将输出值添加到输入向量中
            input.push_back(n->output);
        }

        // 遍历当前层的节点
        for (auto n : l->neurons)
        {
            // 遍历当前层的权重
            for (int j = 0; j < n->weights.size(); ++j)
            {
                // 将权重值乘以输出值，添加到梯度向量中
                grad.push_back(n->delta * input.at(j));
            }
            // 将梯度值添加到梯度向量中
            grad.push_back(n->delta);
        }
        // 清空输入向量
        input.clear();
    }
    // 返回梯度向量
    return grad;
}

void Network::updateWeights(std::vector<double> grad, double learningRate)
{
    // 定义一个变量p，用于记录当前层的神经元数量
    int p = 0;
    // 遍历每一层
    for (int i = 1; i < layers.size(); ++i)
    {
        // 获取当前层
        auto l = layers.at(i);

        // 遍历当前层的每一个神经元
        for (auto n : l->neurons)
        {
            // 遍历每一个神经元的权重
            for (int j = 0; j < n->weights.size(); ++j)
            {
                // 更新神经元的权重
                n->weights[j] += learningRate * grad[p++];
            }
            // 更新神经元的偏置
            n->bias += learningRate * grad[p++];
        }
    }
}

// void Network::updateWeights(std::vector<double> input, double learingRate)
// {
//     for (int i = 0; i < layers.size(); ++i)
//     {
//         Layer *l = layers.at(i);
//         if (i != 0)
//         {
//             input.clear();
//             for (Neuron *n : layers.at(i - 1)->neurons)
//             {
//                 input.push_back(n->output);
//             }
//         }

//         for (Neuron *n : l->neurons)
//         {
//             for (int j = 0; j < n->weights.size(); ++j)
//             {
//                 n->weights[j] += learingRate * n->delta * input.at(j);
//             }
//             n->bias += learingRate * n->delta;
//         }
//     }
// }

void Network::train(std::vector<std::vector<double>> x, std::vector<std::vector<double>> y,
                    int epoches, double learningRate,
                    int batchSize, bool shuffle,
                    std::vector<std::vector<double>> test_input,
                    std::vector<std::vector<double>> test_ouput)
{
    printf("Training Epoches = %d LearningRate = %f BatchSize = %d Shuffle = %d \n", epoches, learningRate, batchSize, shuffle);
    printf("TrainDataSize = %d TestDataSize = %d \n", x.size(), test_input.size());
    if (x.size() != y.size() || test_input.size() != test_ouput.size() || x.size() % batchSize != 0)
    {
        printf("There are some errors in the dataset or parameters\n");
        return;
    }

    std::vector<std::string> log;
    auto start = std::chrono::system_clock::now();

    int train_step = 0;

    for (int epo = 1; epo <= epoches; ++epo)
    {
        // shuffle train_datasets
        if (shuffle)
        {
            shuffleData(x, y);
        }
        int iteration = x.size() / batchSize;

        double totalLoss = 0.0;

        for (int j = 0; j < iteration; ++j)
        {
            std::vector<std::vector<double>> totalGrad;
            for (int i = 0; i < batchSize; ++i)
            {
                auto input = x[j * batchSize + i];
                auto expect = y[j * batchSize + i];

                fprop(input);
                // calculate loss
                double loss = calculateLoss(expect);
                totalLoss += loss;
                bprop(expect);
                // update after batchsize
                std::vector<double> grad = collectGrad();
                totalGrad.push_back(grad);
            }
            std::vector<double> avgGrad;
            for (int i = 0; i < totalGrad[0].size(); ++i)
            {
                double sum = 0.0;
                for (int j = 0; j < totalGrad.size(); ++j)
                {
                    sum += totalGrad[j][i];
                }
                avgGrad.push_back(sum / totalGrad.size());
            }
            updateWeights(avgGrad, learningRate);
            ++train_step;
        }

        double accuracy = -1.0;
        if (test_input.size() > 0 && test_input.size() == test_ouput.size())
        {
            accuracy = test(test_input, test_ouput);
        }

        printf("[%d|%d] TotalLoss %f Accuracy %f \n", epo, epoches, totalLoss, accuracy);
        log.push_back(std::to_string(totalLoss) + " " + std::to_string(accuracy));
    }
    auto end = std::chrono::system_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

    std::string path = "./logs/train_logs.txt";
    saveLogs(path, log);

    printf("End Training, Train Step %ld , Time Cost %.2f s, Save Logs %s \n", train_step, 0.000001 * duration, path.c_str());
}

void Network::fprop(std::vector<double> input)
{
    // 遍历层数
    for (int i = 0; i < layers.size(); ++i)
    {
        // 遍历每一层
        for (int j = 0; j < layers[i]->neurons.size(); ++j)
        {
            // 获取当前层的输出
            auto n = layers[i]->neurons[j];
            // 如果是第一层，输出为输入
            if (i == 0)
            {
                n->output = input[j];
            }
            // 如果不是第一层，获取上一层
            else
            {
                auto preLayer = layers[i - 1];
                // 计算输出
                double sum = n->bias;
                // 遍历上一层的输出
                for (int k = 0; k < preLayer->neurons.size(); ++k)
                {
                    // 计算输出
                    sum += preLayer->neurons[k]->output * n->weights[k];
                }
                // 计算输出
                n->output = activate(sum);
            }
        }
    }
}

std::vector<double> Network::predict(std::vector<double> input)
{
    // 调用fprop函数
    fprop(input);

    // 初始化输出结果
    std::vector<double> output;
    // 遍历最后一层的神经元
    for (int i = 0; i < layers.back()->neurons.size(); ++i)
    {
        // 将最后一层的神经元的输出添加到输出结果中
        output.push_back(layers.back()->neurons.at(i)->output);
    }
    // 返回输出结果
    return output;
}

void Network::printNet()
{
    printf("Network Structure:\n");
    printf("--------------------------------------------\n");
    long long int sum = 0;
    // 遍历层
    for (int i = 0; i < layers.size(); ++i)
    {
        std::string s;
        auto l = layers.at(i);
        // 如果是输入层
        if (i == 0)
        {
            printf("[input_layer] input_size = %d \n", l->neurons.size());
        }
        // 如果是输出层
        else if (i == layers.size() - 1)
        {
            printf("[output_layer] output_size = %d \n", l->neurons.size());
        }
        // 如果是隐藏层
        else
        {
            printf("[hidden_layer] neuron_size = %d \n", l->neurons.size());
        }
        // 计算权重数量和
        sum += l->neurons.size() * (l->neurons[0]->weights.size() + 1);
    }
    printf("--------------------------------------------\n");
    printf("Number of parameters: %lld\n", sum);
    printf("--------------------------------------------\n");
}

double Network::test(std::vector<std::vector<double>> input, std::vector<std::vector<double>> output)
{
    // 计算输入和输出的数量
    int totalNum = input.size();
    int correctNum = 0;
    // 遍历输入和输出
    for (int i = 0; i < totalNum; ++i)
    {
        // 获取输入
        auto x = input.at(i);
        // 获取输出
        auto y = output.at(i);
        // 预测输出
        std::vector<double> z = predict(x);

        // 计算正确的输出
        int a = std::max_element(y.begin(), y.end()) - y.begin();
        int b = std::max_element(z.begin(), z.end()) - z.begin();
        correctNum += (a == b);
    }
    // 返回正确率
    return 1.0 * correctNum / totalNum;
}

void Network::saveModel(std::string path)
{
    // 创建一个字符串数组，用于存储层数
    std::vector<std::string> data;
    // 遍历层数
    for (int i = 0; i < layers.size(); ++i)
    {
        // 获取当前层
        auto l = layers.at(i);
        // 将层数添加到数组中
        data.push_back("L" + std::to_string(l->neurons.size()));
        // 遍历层中的每一个神经元
        for (auto n : l->neurons)
        {
            // 创建一个字符串，用于存储神经元的权重和偏置
            std::string nData = "N" + std::to_string(n->weights.size()) + " ";
            // 遍历神经元的权重
            for (int j = 0; j < n->weights.size(); ++j)
            {
                // 将权重添加到字符串中
                nData += std::to_string(n->weights[j]) + ",";
            }
            // 将偏置添加到字符串中
            nData += std::to_string(n->bias);
            // 将字符串添加到数组中
            data.push_back(nData);
        }
    }
    // 创建一个文件对象
    std::ofstream ofs;
    // 将文件对象打开
    ofs.open(path, std::ios::out);
    // 如果文件对象打开失败，则输出错误信息
    if (!ofs.is_open())
    {
        printf("saveModel Error\n");
        return;
    }
    // 遍历数组
    for (std::string s : data)
    {
        // 将数组中的每一个字符串添加到文件对象中
        ofs << s << std::endl;
    }
    // 关闭文件对象
    ofs.close();
}

void Network::loadModel(std::string path)
{
    printf("LoadModel from %s \n", path.c_str());
    std::ifstream ifs;
    ifs.open(path, std::ios::in);
    if (!ifs.is_open())
    {
        printf("LoadModel Error\n");
        return;
    }
    // 初始化层数组
    layers.clear();

    // 读取每一行
    std::string line;
    while (std::getline(ifs, line))
    {
        // 如果是L，则获取层的数量
        if (line[0] == 'L')
        {
            int size = std::stoi(line.substr(1, line.size() - 1));
            layers.push_back(std::make_shared<Layer>());
        }
        // 如果是N，则获取层的神经元数量
        else if (line[0] == 'N')
        {
            auto l = layers.back();
            int num = 0;
            std::string s;
            std::shared_ptr<Neuron> neu = std::make_shared<Neuron>();
            // 遍历每一个神经元
            for (int i = 1; i < line.size(); ++i)
            {
                // 如果是空格，则获取神经元的数量
                if (line[i] == ' ')
                {
                    num = std::stoi(s);
                    s.clear();
                }
                // 如果是逗号，则获取神经元的权重
                else if (line[i] == ',')
                {
                    neu->weights.push_back(std::stod(s));
                    s.clear();
                }
                // 如果是分号，则获取神经元的偏置
                else
                {
                    s += line[i];
                }
            }
            // 获取神经元的偏置
            neu->bias = std::stod(s);
            l->neurons.push_back(neu);
        }
    }
    // 关闭文件
    ifs.close();
    printf("LoadModel Success \n");
}

Network::~Network()
{
}