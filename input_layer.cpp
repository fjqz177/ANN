#include "input_layer.hpp"

input_layer::input_layer(int size){
    this->preLayer = nullptr;
    for(int i = 0;i<size;++i){
        neurons.push_back(new neuron());
    }
}

input_layer::~input_layer(){
    
}