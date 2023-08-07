#include "layer.hpp"
#include "network.hpp"
#include "dataset.hpp"

#include "data.hpp"

#include <cstdio>
#include <vector>

int main()
{
    printf("Welcome to mechine learning\n");
    // 定义数据集
    DataSet ds;
    // ds.readIrisData();
    ds.readMnistData();

    // 将数据转换为标准形式
    std::vector<std::vector<double>> x = ds.getInput();
    std::vector<std::vector<double>> y = ds.getOutput();

    // 将数据归一化
    std::vector<std::vector<double>> x_test = ds.getTestInput();
    std::vector<std::vector<double>> y_test = ds.getTestOutput();

    // make data normalized
    // 将数据归一化
    std::vector<std::vector<double>> x_n = ds.getNormalizedData(x);
    std::vector<std::vector<double>> y_n = ds.getNormalizedData(y);

    // 将数据归一化
    std::vector<std::vector<double>> x_test_n = ds.getNormalizedData(x_test);
    std::vector<std::vector<double>> y_test_n = ds.getNormalizedData(y_test);

    // 设置神经网络的超参数
    std::vector<int> spec = {28 * 28, 10, 10}; // the num of neuron in every layer
    Network net(spec);
    // 开始训练
    net.train(x_n, y_n, 10, 0.1, 16, true, x_test_n, y_test_n);
    // 保存模型
    net.saveModel("./models/net.model");

    // Network net;
    // net.loadModel("./models/net.model");

    // 测试训练结果
    printf("accuracy %f\n", net.test(x_n, y_n));
    // 打印神经网络的参数
    net.printNet();

    // predict test
    // 打印测试结果
    // for (int i = 1; i < 11; ++i)
    // {
    //     std::vector<double> input = x_n[i];
    //     std::vector<double> expect = y_n[i];

    //     ds.printDigit(input, 0.6);

    //     std::vector<double> output = net.predict(input);
    //     printf("Predict expect is: \n");
    //     printData(expect);
    //     printf("expect is %d\n", maxIndex(expect));
    //     printf("Predict output is: \n");
    //     printData(output);
    //     printf("result is %d\n", maxIndex(output));
    // }

    return 0;
}