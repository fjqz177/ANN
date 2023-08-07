#include "dataset.hpp"
#include <cstdint>

DataSet::DataSet()
{
}

void DataSet::readMnistTrainLable()
{
    // 定义标签
    label = {"0", "1", "2", "3", "4", "5", "6", "7", "8", "9"};
    // 打开文件
    std::ifstream ifsLable;
    ifsLable.open("./datasets/MNIST_data/train-labels.idx1-ubyte", std::ios::in | std::ios::binary);
    // 创建字节数组
    unsigned char bytes[8];
    // 读取文件头
    ifsLable.read((char *)bytes, 8);
    // 获取文件头的魔术数
    uint32_t magic = (uint32_t)((bytes[0] << 24) |
                                (bytes[1] << 16) |
                                (bytes[2] << 8) |
                                bytes[3]);
    // 获取文件头的数量
    uint32_t num = (uint32_t)((bytes[4] << 24) |
                              (bytes[5] << 16) |
                              (bytes[6] << 8) |
                              bytes[7]);
    // 打印文件头信息
    // printf("MnistTrainLable %d %d \n", magic, num);
    // 循环读取数据
    while (!ifsLable.eof())
    {
        // 创建字节数组
        unsigned char byte;
        // 读取字节
        ifsLable.read((char *)&byte, 1);
        // 如果读取失败，则跳出循环
        if (ifsLable.fail())
        {
            break;
        }
        // 获取字节的位置
        int pos = (unsigned int)byte;
        // 创建10个0的vector
        std::vector<double> y(10, 0.0);
        // 将字节转换为double类型
        y[pos] = 1.0;
        // 将y添加到训练输出中
        train_output.push_back(y);
    }
    // 关闭文件
    ifsLable.close();
}

void DataSet::readMnistTestLable()
{
    // 定义标签
    label = {"0", "1", "2", "3", "4", "5", "6", "7", "8", "9"};
    // 打开文件
    std::ifstream ifsLable;
    ifsLable.open("./datasets/MNIST_data/t10k-labels.idx1-ubyte", std::ios::in | std::ios::binary);
    // 定义字节数组
    unsigned char bytes[8];
    // 读取字节数组
    ifsLable.read((char *)bytes, 8);
    // 获取magic和num
    uint32_t magic = (uint32_t)((bytes[0] << 24) |
                                (bytes[1] << 16) |
                                (bytes[2] << 8) |
                                bytes[3]);
    uint32_t num = (uint32_t)((bytes[4] << 24) |
                              (bytes[5] << 16) |
                              (bytes[6] << 8) |
                              bytes[7]);
    // 打印magic和num
    // printf("MnistTestLable %d %d \n", magic, num);
    // 循环读取
    while (!ifsLable.eof())
    {
        // 定义字节
        unsigned char byte;
        // 读取字节
        ifsLable.read((char *)&byte, 1);
        // 如果读取失败，则终止循环
        if (ifsLable.fail())
        {
            break;
        }
        // 将字节转换为int类型
        int pos = (unsigned int)byte;
        // 定义结果
        std::vector<double> y(10, 0.0);
        // 将字节转换为double类型
        y[pos] = 1.0;
        // 将结果添加到test_output中
        test_output.push_back(y);
    }
    // 关闭文件
    ifsLable.close();
}

void DataSet::readMnistTrainImage()
{
    // 打开文件
    std::ifstream ifsLable;
    ifsLable.open("./datasets/MNIST_data/train-images.idx3-ubyte", std::ios::in | std::ios::binary);
    // 声明字节数组
    unsigned char bytes[16];
    // 读取16个字节
    ifsLable.read((char *)bytes, 16);
    // 获取magic值
    uint32_t magic = (uint32_t)((bytes[0] << 24) |
                                (bytes[1] << 16) |
                                (bytes[2] << 8) |
                                bytes[3]);
    // 获取num值
    uint32_t num = (uint32_t)((bytes[4] << 24) |
                              (bytes[5] << 16) |
                              (bytes[6] << 8) |
                              bytes[7]);
    // 获取rows值
    uint32_t rows = (uint32_t)((bytes[8] << 24) |
                               (bytes[9] << 16) |
                               (bytes[10] << 8) |
                               bytes[11]);
    // 获取cols值
    uint32_t cols = (uint32_t)((bytes[12] << 24) |
                               (bytes[13] << 16) |
                               (bytes[14] << 8) |
                               bytes[15]);
    // 打印magic值，num值，rows值，cols值
    // printf("MnistTrainImage %d %d %d %d\n", magic, num, rows, cols);
    // 循环读取数据
    while (!ifsLable.eof())
    {
        int cnt = 0;
        std::vector<double> x;
        // 循环读取数据
        while (cnt < rows * cols && !ifsLable.fail())
        {
            unsigned char byte;
            // 读取字节
            ifsLable.read((char *)&byte, 1);
            // 获取像素值
            int pix = (unsigned int)byte;
            // 将像素值添加到x中
            x.push_back(pix);
            ++cnt;
        }
        // 判断x中是否有数据
        if (x.size() == rows * cols)
            // 如果有数据，将x添加到train_input中
            train_input.push_back(x);
    }
    // 关闭文件
    ifsLable.close();
}

void DataSet::readMnistTestImage()
{
    // 打开文件
    std::ifstream ifsLable;
    ifsLable.open("./datasets/MNIST_data/t10k-images.idx3-ubyte", std::ios::in | std::ios::binary);
    // 读取16个字节
    unsigned char bytes[16];
    ifsLable.read((char *)bytes, 16);
    // 读取magic
    uint32_t magic = (uint32_t)((bytes[0] << 24) |
                                (bytes[1] << 16) |
                                (bytes[2] << 8) |
                                bytes[3]);
    // 读取num
    uint32_t num = (uint32_t)((bytes[4] << 24) |
                              (bytes[5] << 16) |
                              (bytes[6] << 8) |
                              bytes[7]);
    // 读取rows
    uint32_t rows = (uint32_t)((bytes[8] << 24) |
                               (bytes[9] << 16) |
                               (bytes[10] << 8) |
                               bytes[11]);
    // 读取cols
    uint32_t cols = (uint32_t)((bytes[12] << 24) |
                               (bytes[13] << 16) |
                               (bytes[14] << 8) |
                               bytes[15]);
    // 打印magic和num
    // printf("MnistTestImage %d %d %d %d\n", magic, num, rows, cols);
    // 循环读取
    while (!ifsLable.eof())
    {
        int cnt = 0;
        std::vector<double> x;
        while (cnt < rows * cols && !ifsLable.fail())
        while (cnt < rows * cols &&!ifsLable.fail())
        {
            unsigned char byte;
            ifsLable.read((char *)&byte, 1);
            int pix = (unsigned int)byte;
            x.push_back(pix);
            ++cnt;
        }
        if (x.size() == rows * cols)
            test_input.push_back(x);
    }
    // 关闭文件
    ifsLable.close();
}

void DataSet::printDigit(std::vector<double> x, double mask)
{
    // 如果x的长度不等于28*28，则打印错误信息
    if (x.size() != 28 * 28)
    {
        printf("printDigit Error\n");
        return;
    }
    // 遍历每一行
    for (int i = 0; i < 28; ++i)
    {
        // 遍历每一列
        for (int j = 0; j < 28; ++j)
        {
            // 如果x中的值大于mask，则打印##
            if (x[i * 28 + j] > mask)
            {
                printf("##");
            }
            // 否则打印空格
            else
            {
                printf("  ");
            }
        }
        // 换行
        printf("\n");
    }
}

void DataSet::readMnistData()
{
    // 读取训练标签
    readMnistTrainLable();
    // 读取训练图像
    readMnistTrainImage();
    // 读取测试图像
    readMnistTestImage();
    // 读取测试标签
    readMnistTestLable();
    // 打印训练图像和训练标签的大小
    printf("train_image = %d train_lable = %d \n", train_input.size(), train_output.size());
    // 打印测试图像和测试标签的大小
    printf("test_image = %d test_lable = %d \n", test_input.size(), test_output.size());
}

void DataSet::readIrisData()
{
    // 定义标签
    label = {"Iris-setosa", "Iris-versicolor", "Iris-virginica"};
    // 打开文件
    std::ifstream f;
    f.open("./datasets/IRIS_data/iris.data", std::ios::in);
    // 定义一个字符串变量
    std::string line;
    // 循环读取文件中的每一行
    while (std::getline(f, line))
    {
        // 定义一个数组
        std::vector<double> x;
        std::vector<double> y;
        // 定义一个字符串变量
        std::string s;
        // 循环读取每一行中的每一个字符
        for (int i = 0; i < line.size(); ++i)
        {
            // 如果当前字符为逗号，则将字符串变量s赋值为空
            if (line[i] == ',')
            {
                x.push_back(std::stof(s));
                s.clear();
            }
            // 如果当前字符不为逗号，则将字符串变量s加入到字符串变量s中
            else
            {
                s += line[i];
            }
        }
        // 循环读取每一行中的每一个标签
        for (int i = 0; i < label.size(); ++i)
        {
            // 如果当前标签与标签数组中的某一个标签相同，则将1.0赋值给y数组
            if (label[i] == s)
            {
                y.push_back(1.0);
            }
            // 否则，将0.0赋值给y数组
            else
            {
                y.push_back(0.0);
            }
        }

        // 将x数组和y数组放入train_input和train_output数组中
        train_input.push_back(x);
        train_output.push_back(y);
    }
    // 关闭文件
    f.close();
}

DataSet::~DataSet()
{
}

std::vector<std::vector<double>> DataSet::getInput()
{
    return train_input;
}

std::vector<std::vector<double>> DataSet::getOutput()
{
    return train_output;
}

std::vector<std::vector<double>> DataSet::getTestInput()
{
    return test_input;
}

std::vector<std::vector<double>> DataSet::getTestOutput()
{
    return test_output;
}

std::vector<std::vector<double>> DataSet::getNormalizedData(std::vector<std::vector<double>> data)
{
    // 获取归一化后的数据
    auto getNormalized = [](double d, double min, double max)
    {
        // 如果最大值和最小值相等，则直接返回
        if (max == min)
            return d;
        // 返回归一化后的值
        return (d - min) / (max - min);
    };

    // 如果数据集为空，则直接返回
    if (data.size() == 0)
    {
        return data;
    }

    // 获取最大值和最小值
    std::vector<double> maxVec = data[0];
    std::vector<double> minVec = data[0];
    for (int i = 0; i < data.size(); ++i)
    {
        // 获取每一列的最大值和最小值
        for (int j = 0; j < maxVec.size(); ++j)
        {
            maxVec[j] = std::max(maxVec[j], data[i][j]);
            minVec[j] = std::min(minVec[j], data[i][j]);
        }
    }
    // 创建一个空的x数组
    std::vector<std::vector<double>> x;

    // 将每一行的数据转换成x数组
    for (int i = 0; i < data.size(); ++i)
    {
        std::vector<double> item;
        // 遍历每一列
        for (int j = 0; j < maxVec.size(); ++j)
        {
            // 获取归一化后的值
            item.push_back(getNormalized(data[i][j], minVec[j], maxVec[j]));
        }
        x.push_back(item);
    }

    // 返回归一化后的x数组
    return x;
}
