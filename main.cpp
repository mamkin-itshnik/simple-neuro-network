#include <QCoreApplication>
#include <QDebug>
#include <QTime>
#include "myneuro.h"
#include <QFile>

float* inputs_list(const QStringList &strList)
{
    float* inputs = (float*) malloc((784)*sizeof(float));
    QString str;
    bool ok=true;
    for (int i = 1; i<strList.size();i++)
    {
        str = strList.at(i);
        inputs[i-1]= ( (str.toFloat(&ok) / 255.0 *0.99)+0.01);
    }
    return inputs;
}

float* targets_list(const int &j)
{
     float* targets = (float*) malloc((10)*sizeof(float));
    for (int i = 0; i<10;i++)
    {
        if(i==j)
        targets[i]=(0.99);
        else
        targets[i]=(0.01);
    }

    return targets;
}


int main(int argc, char *argv[])
{
    QCoreApplication a(argc, argv);

    myNeuro *nW = new myNeuro();
    QStringList wordList;
    bool ok=true;
        QFile f("/home/evgen/GitHub/simple-neuro-network/mnist_train.csv");
        if (f.open(QIODevice::ReadOnly))
        {
            int qq=0;
//            while(!f.atEnd())
           while(qq<3000)
            {
                qq++;
                if(qq%100==0)
                    qDebug()<<qq;
                QString data;
                data = f.readLine();
                wordList = data.split(',');
                QString str = wordList.at(0);
                float * tmpIN = inputs_list(wordList);
                float * tmpTAR = targets_list(str.toInt(&ok));
                nW->train(tmpIN,tmpTAR);
                delete tmpIN;
                delete tmpTAR;
            }

            f.close();
        }
        QFile f2("/home/evgen/GitHub/simple-neuro-network/mnist_test_10.csv");
        if (f2.open(QIODevice::ReadOnly))
        {
            while(!f2.atEnd())
            {
                QString data;
                data = f2.readLine();
                wordList = data.split(',');
                QString str = wordList.at(0);
                qDebug()<<"__________________";
                qDebug()<<"For number "<<str;
                float * tmpIN = inputs_list(wordList);
                nW->query(tmpIN);
                delete tmpIN;
                tmpIN = nullptr;
            }

            f2.close();
        }
         delete nW;
        nW = nullptr;
        qDebug()<<"_______________THE____END_______________";



    return a.exec();
}
