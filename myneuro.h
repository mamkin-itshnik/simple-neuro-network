#ifndef MYNEURO_H
#define MYNEURO_H
#include <math.h>
#include <QtGlobal>
#include <vector>
using namespace std;

#define learnRate 0.5
#define randWeight (( ((float)qrand() / (float)RAND_MAX) - 0.5)* pow(out,-0.5))
class myNeuro
{
public:
    myNeuro();
    ~myNeuro();
	
    struct nnLay{

            //--- information about input/out width for neuro layer
           int in;
           int out;
           //--- weight matrix
           float** matrix;
           //--- current hidden value array
           float* hidden;
           //--- current errors for backPropagate
           float* errors;
           int getInCount(){return in;}
           int getOutCount(){return out;}
           float **getMatrix(){return matrix;}
           void updMatrix(float *enteredVal)
           {
               //--- upd weight with considering errors
               for(int ou =0; ou < out; ou++)
               {

                   for(int hid =0; hid < in; hid++)
                   {
                       matrix[hid][ou] += (learnRate * errors[ou] * enteredVal[hid]);
                   }
                   matrix[in][ou] += (learnRate * errors[ou]);
               }
           };
           void setIO(int inputs, int outputs)
           {
               //--- initialization values and allocating memory
               in=inputs;
               out=outputs;
               errors = (float*) malloc((out)*sizeof(float));
               hidden = (float*) malloc((out)*sizeof(float));

               matrix = (float**) malloc((in+1)*sizeof(float*));
               for(int inp =0; inp < in+1; inp++)
               {
                   matrix[inp] = (float*) malloc(out*sizeof(float));
               }
               for(int inp =0; inp < in+1; inp++)
               {
                   for(int outp =0; outp < out; outp++)
                   {
                       matrix[inp][outp] =  randWeight;
                   }
               }
           }
           void makeHidden(float *inputs)
           {
               //--- make value after signal passing current layer
               for(int hid =0; hid < out; hid++)
               {
                   float tmpS = 0.0;
                   for(int inp =0; inp < in; inp++)
                   {
                       tmpS += inputs[inp] * matrix[inp][hid];
                   }
                   tmpS += matrix[in][hid];
                   hidden[hid] = sigmoida(tmpS);
               }
           };
           float* getHidden()
           {
               return hidden;
           };
           void calcOutError(float *targets)
           {
               //--- calculating error if layer is last
               for(int ou =0; ou < out; ou++)
               {
                   errors[ou] = (targets[ou] - hidden[ou]) * sigmoidasDerivate(hidden[ou]);
               }
           };
           void calcHidError(float *targets,float **outWeights,int inS, int outS)
           {
               //--- calculating error if layer is hidden
               for(int hid =0; hid < inS; hid++)
               {
                   errors[hid] = 0.0;
                   for(int ou =0; ou < outS; ou++)
                   {
                       errors[hid] += targets[ou] * outWeights[hid][ou];
                   }
                   errors[hid] *= sigmoidasDerivate(hidden[hid]);
               }
           };
           float* getErrors()
           {
               return errors;
           };
           float sigmoida(float val)
           {
               //--- activation function
              return (1.0 / (1.0 + exp(-val)));
           }
           float sigmoidasDerivate(float val)
           {
               //--- activation function derivative
                return (val * (1.0 - val));
           };
    };

    void feedForwarding(bool ok);
    void backPropagate();
    void train(float *in, float *targ);
    void query(float *in);
    void printArray(float *arr,int s);

private:
    std::vector<nnLay> *_nList = nullptr;
    int _inputNeurons;
    int _outputNeurons;
    int _nlCount;

    float *_inputs = nullptr;
    float *_targets = nullptr;
};

#endif // MYNEURO_H
