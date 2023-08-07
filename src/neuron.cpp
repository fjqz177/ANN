#include "neuron.hpp"

Neuron::Neuron()
{
    // 生成随机数
    srand((unsigned int)time(NULL));

    // 初始化偏置
    bias = 0;
    // 初始化输出
    delta = 0;
    // 初始化输出
    output = 0;
}

// 生成随机数
double Neuron::generateRandom(double min, double max)
{
    // 生成随机数
    double rd = (double)rand() / RAND_MAX;
    // 返回min和max之间的随机数
    return min + rd * (max - min);
}

Neuron::Neuron(int preSize)
{
    // 生成随机数
    srand((unsigned int)time(NULL));

    // 生成偏置
    bias = generateRandom(-1, 1);
    // 遍历前面的神经元
    for (int i = 0; i < preSize; ++i)
    {
        // 生成权重
        weights.push_back(generateRandom(-1, 1));
    }
    // 初始化输出
    delta = 0.0;
    // 初始化输出
    output = 0.0;
}

Neuron::~Neuron()
{
}
