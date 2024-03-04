//
// Created by tomma on 03/03/2024.
//

#include "MainNode.h"
#include "FPGAMatrixVectorMultiplier.h"
#include "AcceleratorNode.h"

int main() {
    MPI_Init(nullptr, nullptr);
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if(rank == 0) {
        MainNode<FPGAMatrixVectorMultiplier> mainNode("", "", 100, 1e-12);
        mainNode.init();
        mainNode.handshake();
        mainNode.compute_conjugate_gradient();
    } else {
        AcceleratorNode<FPGAMatrixVectorMultiplier> acceleratorNode;
        acceleratorNode.init();
        acceleratorNode.handshake();
        acceleratorNode.compute();
    }



}
