//
// Created by tomma on 03/03/2024.
//

#include "FpgaAcceleratorNode.h"
#include "MainNode.h"

int main() {
    MPI_Init(nullptr, nullptr);
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if(rank == 0) {
        MainNode mainNode("", "", 1000, 1e-12);
        mainNode.handshake();
    } else {
        FpgaAcceleratorNode fpgaNode;
        //fpgaNode.setup();
        fpgaNode.handshake();
    }



}
