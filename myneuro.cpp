#include "myneuro.h"
#include <QDebug>

myNeuro::myNeuro()
{
//    //--------многослойный
//    inputNeurons = 100;
//    outputNeurons =2;
//    nlCount = 4;
//    list = (nnLay*) malloc((nlCount)*sizeof(nnLay));

//    inputs = (float*) malloc((inputNeurons)*sizeof(float));
//    targets = (float*) malloc((outputNeurons)*sizeof(float));

//    list[0].setIO(100,20);
//    list[1].setIO(20,6);
//    list[2].setIO(6,3);
//    list[3].setIO(3,2);

    //--------однослойный---------
    inputNeurons = 784;
    outputNeurons =10;
    nlCount = 2;
    list = (nnLay*) malloc((nlCount)*sizeof(nnLay));

    inputs = (float*) malloc((inputNeurons)*sizeof(float));
    targets = (float*) malloc((outputNeurons)*sizeof(float));

    list[0].setIO(784,200);
    list[1].setIO(200,10);

}

void myNeuro::feedForwarding(bool ok)
{
    list[0].makeHidden(inputs);
    for (int i =1; i<nlCount; i++)
        list[i].makeHidden(list[i-1].getHidden());

    if (!ok)
    {
        qDebug()<<"Feed Forward: ";
        for(int out =0; out < outputNeurons; out++)
        {
            qDebug()<<list[nlCount-1].hidden[out];
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
    //-------------------------------ERRORS-----CALC---------
    list[nlCount-1].calcOutError(targets);
    for (int i =nlCount-2; i>=0; i--)
        list[i].calcHidError(list[i+1].getErrors(),list[i+1].getMatrix(),
                list[i+1].getInCount(),list[i+1].getOutCount());

    //-------------------------------UPD-----WEIGHT---------
    for (int i =nlCount-1; i>0; i--)
        list[i].updMatrix(list[i-1].getHidden());
    list[0].updMatrix(inputs);
}

void myNeuro::train(float *in, float *targ)
{
    inputs = in;
    targets = targ;
    feedForwarding(true);
}

void myNeuro::query(float *in)
{
    inputs=in;
    feedForwarding(false);
}

void myNeuro::printArray(float *arr, int s)
{
    qDebug()<<"__";
    for(int inp =0; inp < s; inp++)
    {
        qDebug()<<arr[inp];
    }
}