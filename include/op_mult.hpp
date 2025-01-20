#ifndef OP_MULT_HPP
#define OP_MULT_HPP

#include <cudnn.h>

template <typename Image_T>
class OpMult {
   public:
    OpMult(float beta = 0.0f) : m_beta(beta) {}

    void apply(Image_T& output, const Image_T& input1, const Image_T& input2) const {
        assert(input1.width() == output.width());
        assert(input1.height() == output.height());
        assert(input2.width() == output.width());
        assert(input2.height() == output.height());

        cudnnHandle_t cudnn;
        CHECK_CUDNN(cudnnCreate(&cudnn));

        // Define input tensor descriptor
        cudnnTensorDescriptor_t tensorDesc;
        CHECK_CUDNN(cudnnCreateTensorDescriptor(&tensorDesc));
        CHECK_CUDNN(cudnnSetTensor4dDescriptor(tensorDesc, CUDNN_TENSOR_NHWC, CUDNN_DATA_FLOAT, 1,
                                               Image_T::channels(), input1.height(),
                                               input1.width()));

        // Create an OpTensor descriptor
        cudnnOpTensorDescriptor_t opTensorDesc;
        cudnnCreateOpTensorDescriptor(&opTensorDesc);
        cudnnSetOpTensorDescriptor(opTensorDesc, CUDNN_OP_TENSOR_MUL, CUDNN_DATA_FLOAT,
                                   CUDNN_PROPAGATE_NAN);

        const float alpha = 1.0f;
        // Perform the squaring operation
        cudnnOpTensor(cudnn, opTensorDesc, &alpha, tensorDesc, input1.data(), &alpha, tensorDesc,
                      input2.data(), &m_beta, tensorDesc, output.data());

        // Cleanup
        cudnnDestroyOpTensorDescriptor(opTensorDesc);
        cudnnDestroyTensorDescriptor(tensorDesc);
        cudnnDestroy(cudnn);
    }

   private:
    float m_beta;
};

#endif  // OP_MULT_HPP