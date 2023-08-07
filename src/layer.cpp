#include "layer.hpp"

Layer::Layer()
{
}

Layer::~Layer()
{
}

// 创建一个Layer对象，preSize为神经元的数量，size为Layer的大小
Layer::Layer(size_t preSize, size_t size)
{
    // 循环创建Layer的神经元
    for (size_t i = 0; i < size; ++i)
    {
        // 创建一个神经元，preSize为神经元的数量
        neurons.push_back(std::make_shared<Neuron>(preSize));
    }
}
