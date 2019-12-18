#include "myneuro.h"
#include <stdlib.h>
#include <iostream>

myNeuro::myNeuro()
{
    //--- "Neyeral Network" this equal "NN"
    //---set width for NN input
    _inputNeurons = 784;
    //---set width for NN output
    _outputNeurons =10;
    //---set layer count for NN,
    //---where input neuerons for first layer equal NN input
    //---and output neuerons for last layer equal NN output
    _nlCount = 2;
    _nList = new vector<nnLay>(_nlCount);

    //--- set input and output array size for every layer
    //--- where first layer have "NN input" input size
    //--- and last layer have "NN output" output size
    _nList->at(0).setIO(_inputNeurons,200);
    _nList->at(1).setIO(200,_outputNeurons);


    //--------- examples for more layer
    //--------- WARNING!!!!! -------input size for every layer
    //------------------------------must equals output size previous layer
    /*
    _nlCount = 3;
        _nList = new vector<nnLay>(_nlCount);

        _nList -> at(0).setIO(_inputNeurons, 200);
        _nList -> at(1).setIO(200, 60);
        _nList -> at(2).setIO(60, _outputNeurons);

    //------- examples for more layer
    _nlCount = 4;
        _nList = new vector<nnLay>(_nlCount);

        _nList -> at(0).setIO(_inputNeurons, 200);
        _nList -> at(1).setIO(200, 60);
        _nList -> at(2).setIO(60, 40);
       _nList -> at(3).setIO(40, _outputNeurons);
       */
}

myNeuro::~myNeuro()
{
    for (int i =0; i<_nlCount; i++)
    {
        delete []_nList->data()[i].hidden;
        _nList->data()[i].hidden = nullptr;

        delete []_nList->data()[i].errors;
        _nList->data()[i].errors = nullptr;

        for(int inp = 0; inp < _nList->data()[i].in; inp++)
        {
            delete []_nList->data()[i].matrix[inp];
            _nList->data()[i].matrix[inp] = nullptr;
        }
        delete []_nList->data()[i].matrix;
        _nList->data()[i].matrix = nullptr;
    }
}

void myNeuro::feedForwarding(bool ok)
{
    //--- signal through NN in forward direction

    //--- for first layer argument is _inputs
    _nList->at(0).makeHidden(_inputs);
   //--- for other layer argument is "hidden" array previous's layer
    for (int i = 1; i<_nlCount; i++)
        _nList->at(i).makeHidden(_nList->at(i-1).getHidden());


    //--- bool condition for query NN or train NN
    if (!ok)
    {
        cout<<"Feed Forward: "<<endl;
        for(int out =0; out < _outputNeurons; out++)
        {
            cout<<_nList->at(_nlCount-1).hidden[out]<<endl;
        }
        return;
    }
    else
    {
        // printArray(list[3].getErrors(),list[3].getOutCount());
        backPropagate();
    }
}

void myNeuro::backPropagate()
{   
    //--- calculate errors for last layer
    _nList->at(_nlCount-1).calcOutError(_targets);
    //--- for others layers to calculate errors we need information about "next layer"
    //---   //for example// to calculate 4'th layer errors we need 5'th layer errors
    for (int i = _nlCount-2; i>=0; i--)
        _nList->at(i).calcHidError(
                                   _nList->at(i+1).getErrors(),
                                   _nList->at(i+1).getMatrix(),
                                   _nList->at(i+1).getInCount(),
                                   _nList->at(i+1).getOutCount()
                                   );

    //--- updating weights
    //--- to UPD weight for current layer we must get "hidden" value array of previous layer
    for (int i = _nlCount-1; i>0; i--)
        _nList->at(i).updMatrix(_nList->at(i-1).getHidden());
    //--- first layer hasn't previous layer.
    //--- for him "hidden" value array of previous layer be NN input
    _nList->at(0).updMatrix(_inputs);
}

void myNeuro::train(float *in, float *targ)
{
    if(in)
    _inputs = in;
    if(targ)
    _targets = targ;

    //--- bool == true enable backPropogate function, else it's equal query without print
    feedForwarding(true);
}

void myNeuro::query(float *in)
{
    _inputs = in;
    //--- bool == false call query NN with print NN output
    feedForwarding(false);
}

void myNeuro::printArray(float *arr, int s)
{
    cout<<"__"<<endl;
    for(int inp =0; inp < s; inp++)
    {
        cout<<arr[inp]<<endl;
    }
}
