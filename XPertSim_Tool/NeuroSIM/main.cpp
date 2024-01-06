/*******************************************************************************
* Copyright (c) 2015-2017
* School of Electrical, Computer and Energy Engineering, Arizona State University
* PI: Prof. Shimeng Yu
* All rights reserved.
* 
* This source code is part of NeuroSim - a device-circuit-algorithm framework to benchmark 
* neuro-inspired architectures with synaptic devices(e.g., SRAM and emerging non-volatile memory). 
* Copyright of the model is maintained by the developers, and the model is distributed under 
* the terms of the Creative Commons Attribution-NonCommercial 4.0 International Public License 
* http://creativecommons.org/licenses/by-nc/4.0/legalcode.
* The source code is free and you can redistribute and/or modify it
* by providing that the following conditions are met:
* 
*  1) Redistributions of source code must retain the above copyright notice,
*     this list of conditions and the following disclaimer.
* 
*  2) Redistributions in binary form must reproduce the above copyright notice,
*     this list of conditions and the following disclaimer in the documentation
*     and/or other materials provided with the distribution.
* 
* THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
* ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
* WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
* DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
* FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
* DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
* SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
* CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
* OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
* OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
* 
* Developer list: 
*   Pai-Yu Chen	    Email: pchen72 at asu dot edu 
*                    
*   Xiaochen Peng   Email: xpeng15 at asu dot edu
********************************************************************************/

#include <cstdio>
#include <random>
#include <cmath>
#include <iostream>
#include <fstream>
#include <string>
#include <stdlib.h>
#include <vector>
#include <sstream>
#include <chrono>
#include <algorithm>
#include "constant.h"
#include "formula.h"
#include "Param.h"
#include "Tile.h"
#include "Chip.h"
#include "ProcessingUnit.h"
#include "SubArray.h"
#include "Definition.h"

using namespace std;

vector<vector<double> > getNetStructure(const string &inputfile);
vector <double> ela_calc(vector<vector<double> > &netStructure, vector<string> &weights, vector<string> &inputs, int la);

vector <double> ela_calc(vector<vector<double> > &netStructure, vector<string> &weights, vector<string> &inputs, int la) {
//
	auto start = chrono::high_resolution_clock::now();
//
	gen.seed(0);

//	vector<vector<double> > netStructure;
//	netStructure = getNetStructure(argv[1]);

	// define weight/input/memory precision from wrapper
	param->synapseBit = 8;              // precision of synapse weight
//	param->numBitInput = 4;            // precision of input neural activation
//    param->numColMuxed = netStructure[0][8];
//    param->levelOutput = netStructure[0][9];
//    param->SARADC = false;

    param->numBitInput = netStructure[0][10];
	param->levelOutput = netStructure[0][11];           // precision of input neural activation
    param->numColMuxed = netStructure[0][8];
    if (netStructure[0][9] == 1)
        param->SARADC = false;
    else
        param->SARADC = true;
//    if (netStructure[0][9] == 1)
//        param->SARADC = false;
//    else
//        param->SARADC = true;

//    if (la<2)
//        param->levelOutput = 32;
//    else
//        param->levelOutput = 64;
    cout << "DEBUG WEIGHTS AND INPUT FILES" << weights[0] << weights[1]<< inputs[0] << inputs[1] << param->numColMuxed << param->SARADC << endl;

//    mux = netStructure[i][7];
//    adc = netStructure[i][7];
//    if (atoi(argv[5]) == 0){
//        param->SARADC = false;
//    }
//    else
//        param->SARADC = true;
//
//
//
//    cout << "SARADC   "<< param->SARADC << endl;
////	param->numBitInput = 8;
//
	if (param->cellBit > param->synapseBit) {
		cout << "ERROR!: Memory precision is even higher than synapse precision, please modify 'cellBit' in Param.cpp!" << endl;
		param->cellBit = param->synapseBit;
	}

	/*** initialize operationMode as default ***/
	param->conventionalParallel = 0;
	param->conventionalSequential = 0;
	param->BNNparallelMode = 0;                // parallel BNN
	param->BNNsequentialMode = 0;              // sequential BNN
	param->XNORsequentialMode = 0;           // Use several multi-bit RRAM as one synapse
	param->XNORparallelMode = 0;         // Use several multi-bit RRAM as one synapse
	switch(param->operationmode) {
		case 6:	    param->XNORparallelMode = 1;               break;
		case 5:	    param->XNORsequentialMode = 1;             break;
		case 4:	    param->BNNparallelMode = 1;                break;
		case 3:	    param->BNNsequentialMode = 1;              break;
		case 2:	    param->conventionalParallel = 1;           break;
		case 1:	    param->conventionalSequential = 1;         break;
		case -1:	break;
		default:	exit(-1);
	}

	if (param->XNORparallelMode || param->XNORsequentialMode) {
		param->numRowPerSynapse = 2;
	} else {
		param->numRowPerSynapse = 1;
	}
	if (param->BNNparallelMode) {
		param->numColPerSynapse = 2;
	} else if (param->XNORparallelMode || param->XNORsequentialMode || param->BNNsequentialMode) {
		param->numColPerSynapse = 1;
	} else {
		param->numColPerSynapse = ceil((double)param->synapseBit/(double)param->cellBit);
	}
	for (int i =0 ; i < netStructure.size(); i++){
	    for (int j = 0; j < netStructure[i].size(); j++){
	        cout << netStructure[i][j] << ",";
	    }
	    cout << endl;
	}
	double maxPESizeNM, maxTileSizeCM, numPENM;
	vector<int> markNM;
	vector<int> pipelineSpeedUp;
	markNM = ChipDesignInitialize(inputParameter, tech, cell, false, netStructure, &maxPESizeNM, &maxTileSizeCM, &numPENM);
	pipelineSpeedUp = ChipDesignInitialize(inputParameter, tech, cell, true, netStructure, &maxPESizeNM, &maxTileSizeCM, &numPENM);

	double desiredNumTileNM, desiredPESizeNM, desiredNumTileCM, desiredTileSizeCM, desiredPESizeCM;
	int numTileRow, numTileCol;

	vector<vector<double> > numTileEachLayer;
	vector<vector<double> > utilizationEachLayer;
	vector<vector<double> > speedUpEachLayer;
	vector<vector<double> > tileLocaEachLayer;

	numTileEachLayer = ChipFloorPlan(true, false, false, netStructure, markNM,
					maxPESizeNM, maxTileSizeCM, numPENM, pipelineSpeedUp,
					&desiredNumTileNM, &desiredPESizeNM, &desiredNumTileCM, &desiredTileSizeCM, &desiredPESizeCM, &numTileRow, &numTileCol);

	utilizationEachLayer = ChipFloorPlan(false, true, false, netStructure, markNM,
					maxPESizeNM, maxTileSizeCM, numPENM, pipelineSpeedUp,
					&desiredNumTileNM, &desiredPESizeNM, &desiredNumTileCM, &desiredTileSizeCM, &desiredPESizeCM, &numTileRow, &numTileCol);

	speedUpEachLayer = ChipFloorPlan(false, false, true, netStructure, markNM,
					maxPESizeNM, maxTileSizeCM, numPENM, pipelineSpeedUp,
					&desiredNumTileNM, &desiredPESizeNM, &desiredNumTileCM, &desiredTileSizeCM, &desiredPESizeCM, &numTileRow, &numTileCol);

	tileLocaEachLayer = ChipFloorPlan(false, false, false, netStructure, markNM,
					maxPESizeNM, maxTileSizeCM, numPENM, pipelineSpeedUp,
					&desiredNumTileNM, &desiredPESizeNM, &desiredNumTileCM, &desiredTileSizeCM, &desiredPESizeCM, &numTileRow, &numTileCol);

	// for (int i = 0; i < tileLocaEachLayer[0].size(); i++) {
 //            // cout << "num_tile" << tileLocaEachLayer[0][i] << " ";
 //        }
 //    // cout << endl;
 //    for (int i = 0; i < tileLocaEachLayer[1].size(); i++) {
 //            cout << "num_tile" << tileLocaEachLayer[1][i] << " ";
 //        }
    ofstream myfile;
    myfile.open ("./example.txt");
    cout << endl;
	cout << "------------------------------ FloorPlan --------------------------------" <<  endl;
	myfile << "------------------------------ FloorPlan --------------------------------" <<  endl;
	cout << endl;
	myfile << endl;
	cout << "Tile and PE size are optimized to maximize memory utilization ( = memory mapped by synapse / total memory on chip)" << endl;
	myfile << "Tile and PE size are optimized to maximize memory utilization ( = memory mapped by synapse / total memory on chip)" << endl;
	cout << endl;
	myfile << endl;
	if (!param->novelMapping) {
		cout << "Desired Conventional Mapped Tile Storage Size: " << desiredTileSizeCM << "x" << desiredTileSizeCM << endl;
        myfile << "Desired Conventional Mapped Tile Storage Size: " << desiredTileSizeCM << "x" << desiredTileSizeCM << endl;

		cout << "Desired Conventional PE Storage Size: " << desiredPESizeCM << "x" << desiredPESizeCM << endl;
        myfile << "Desired Conventional PE Storage Size: " << desiredPESizeCM << "x" << desiredPESizeCM << endl;

	} else {
		cout << "Desired Conventional Mapped Tile Storage Size: " << desiredTileSizeCM << "x" << desiredTileSizeCM << endl;
		myfile << "Desired Conventional Mapped Tile Storage Size: " << desiredTileSizeCM << "x" << desiredTileSizeCM << endl;
		cout << "Desired Conventional PE Storage Size: " << desiredPESizeCM << "x" << desiredPESizeCM << endl;
		myfile << "Desired Conventional PE Storage Size: " << desiredPESizeCM << "x" << desiredPESizeCM << endl;
		cout << "Desired Novel Mapped Tile Storage Size: " << numPENM << "x" << desiredPESizeNM << "x" << desiredPESizeNM << endl;
		myfile << "Desired Novel Mapped Tile Storage Size: " << numPENM << "x" << desiredPESizeNM << "x" << desiredPESizeNM << endl;
	}
	cout << "User-defined SubArray Size: " << param->numRowSubArray << "x" << param->numColSubArray << endl;
	myfile << "User-defined SubArray Size: " << param->numRowSubArray << "x" << param->numColSubArray << endl;
	cout << endl;
	myfile << endl;
	cout << "----------------- # of tile used for each layer -----------------" <<  endl;
	myfile << "----------------- # of tile used for each layer -----------------" <<  endl;
	double totalNumTile = 0;
	for (int i=0; i<netStructure.size(); i++) {
		cout << "layer" << i+1 << ": " << numTileEachLayer[0][i] * numTileEachLayer[1][i] << endl;
		myfile << "layer" << i+1 << ": " << numTileEachLayer[0][i] * numTileEachLayer[1][i] << endl;
		totalNumTile += numTileEachLayer[0][i] * numTileEachLayer[1][i];
	}
	cout << endl;
	myfile << endl;

	cout << "----------------- Speed-up of each layer ------------------" <<  endl;
	myfile << "----------------- Speed-up of each layer ------------------" <<  endl;
	for (int i=0; i<netStructure.size(); i++) {
		cout << "layer" << i+1 << ": " << speedUpEachLayer[0][i] * speedUpEachLayer[1][i] << endl;
		myfile << "layer" << i+1 << ": " << speedUpEachLayer[0][i] * speedUpEachLayer[1][i] << endl;
	}
	cout << endl;
	myfile << endl;

	cout << "----------------- Utilization of each layer ------------------" <<  endl;
	myfile << "----------------- Utilization of each layer ------------------" <<  endl;
	double realMappedMemory = 0;
	for (int i=0; i<netStructure.size(); i++) {
		cout << "layer" << i+1 << ": " << utilizationEachLayer[i][0] << endl;
		myfile << "layer" << i+1 << ": " << utilizationEachLayer[i][0] << endl;
		realMappedMemory += numTileEachLayer[0][i] * numTileEachLayer[1][i] * utilizationEachLayer[i][0];
	}
	cout << "Memory Utilization of Whole Chip: " << realMappedMemory/totalNumTile*100 << " % " << endl;
	myfile << "Memory Utilization of Whole Chip: " << realMappedMemory/totalNumTile*100 << " % " << endl;
	cout << endl;
	myfile << endl;
	cout << "---------------------------- FloorPlan Done ------------------------------" <<  endl;
	myfile << "---------------------------- FloorPlan Done ------------------------------" <<  endl;
	cout << endl;
	myfile << endl;
	cout << endl;
	myfile << endl;
	cout << endl;
	myfile << endl;

	double numComputation = 0;
	for (int i=0; i<netStructure.size()-1; i++) {
		numComputation += 2*(netStructure[i][0] * netStructure[i][1] * netStructure[i][2] * netStructure[i][3] * netStructure[i][4] * netStructure[i][5]);
	}
//    cout << "before This" << endl;
	ChipInitialize(inputParameter, tech, cell, netStructure, markNM, numTileEachLayer,
					numPENM, desiredNumTileNM, desiredPESizeNM, desiredNumTileCM, desiredTileSizeCM, desiredPESizeCM, numTileRow, numTileCol);

	double chipHeight, chipWidth, chipArea, chipAreaIC, chipAreaADC, chipAreaAccum, chipAreaOther, chipAreaArray;
	double CMTileheight = 0;
	double CMTilewidth = 0;
	double NMTileheight = 0;
	double NMTilewidth = 0;
	vector<double> chipAreaResults;

	chipAreaResults = ChipCalculateArea(inputParameter, tech, cell, desiredNumTileNM, numPENM, desiredPESizeNM, desiredNumTileCM, desiredTileSizeCM, desiredPESizeCM, numTileRow,
					&chipHeight, &chipWidth, &CMTileheight, &CMTilewidth, &NMTileheight, &NMTilewidth);
	chipArea = chipAreaResults[0];
	chipAreaIC = chipAreaResults[1];
	chipAreaADC = chipAreaResults[2];
	chipAreaAccum = chipAreaResults[3];
	chipAreaOther = chipAreaResults[4];
	chipAreaArray = chipAreaResults[5];

	double clkPeriod = 0;
	double layerclkPeriod = 0;

	double chipReadLatency = 0;
	double chipReadDynamicEnergy = 0;
	double chipLeakageEnergy = 0;
	double chipLeakage = 0;
	double chipbufferLatency = 0;
	double chipbufferReadDynamicEnergy = 0;
	double chipicLatency = 0;
	double chipicReadDynamicEnergy = 0;

	double chipLatencyADC = 0;
	double chipLatencyAccum = 0;
	double chipLatencyOther = 0;
	double chipEnergyADC = 0;
	double chipEnergyAccum = 0;
	double chipEnergyOther = 0;

	double layerReadLatency = 0;
	double layerReadDynamicEnergy = 0;
	double tileLeakage = 0;
	double layerbufferLatency = 0;
	double layerbufferDynamicEnergy = 0;
	double layericLatency = 0;
	double layericDynamicEnergy = 0;

	double coreLatencyADC = 0;
	double coreLatencyAccum = 0;
	double coreLatencyOther = 0;
	double coreEnergyADC = 0;
	double coreEnergyAccum = 0;
	double coreEnergyOther = 0;

    vector <double> ela_perf;
    ela_perf.push_back(numComputation);
	if (param->synchronous){
		// calculate clkFreq
		for (int i=0; i<netStructure.size(); i++) {
//
//		    if (i == 0){
//		        cout << "i" << i << endl;
//		        param->numBitInput = 2;
//		        param->SARADC = true;
//		        param->numColMuxed = 4;
//		        }
//            if (i == 1){
//                cout << "i" << i << endl;
//		        param->numBitInput = 2;
//		        }
//            if (i == 2){
//                cout << "i" << i << endl;
//		        param->numBitInput = 2;
//		        }
//
//            if (i == 3){
//                cout << "i" << i << endl;
//		        param->numBitInput = 2;
//		        }
//
//            if (i == 4){
//                cout << "i" << i << endl;
//		        param->numBitInput = 2;
//		        }
//
//            if (i == 5){
//                cout << "i" << i << endl;
//		        param->numBitInput = 2;
//		        }
//
//            if (i == 6){
//                cout << "i" << i << endl;
//		        param->numBitInput = 2;
//		        }
//
//            if (i == 7){
//                cout << "i" << i << endl;
//		        param->numBitInput = 2;
//		        }
//
//            if (i == 8){
//                cout << "i" << i << endl;
//		        param->numBitInput = 2;
//		        }
//
//            if (i == 9){
//                cout << "i" << i << endl;
//		        param->numBitInput = 2;
//		        }
//            if (i == 10){
//                cout << "i" << i << endl;
//		        param->numBitInput = 2;
//		        }
//            if (i == 11){
//                cout << "i" << i << endl;
//		        param->numBitInput = 2;
//		        }



//		    cout << "synch true new" << endl;
			ChipCalculatePerformance(inputParameter, tech, cell, i, weights[i], weights[i], inputs[i], netStructure[i][6],
						netStructure, markNM, numTileEachLayer, utilizationEachLayer, speedUpEachLayer, tileLocaEachLayer,
						numPENM, desiredPESizeNM, desiredTileSizeCM, desiredPESizeCM, CMTileheight, CMTilewidth, NMTileheight, NMTilewidth,
						&layerReadLatency, &layerReadDynamicEnergy, &tileLeakage, &layerbufferLatency, &layerbufferDynamicEnergy, &layericLatency, &layericDynamicEnergy,
						&coreLatencyADC, &coreLatencyAccum, &coreLatencyOther, &coreEnergyADC, &coreEnergyAccum, &coreEnergyOther, true, &layerclkPeriod);
			if(clkPeriod < layerclkPeriod){
				clkPeriod = layerclkPeriod;
			}
		}
		if(param->clkFreq > 1/clkPeriod){
			param->clkFreq = 1/clkPeriod;
		}
	}

	cout << "-------------------------------------- Hardware Performance --------------------------------------" <<  endl;
	myfile << "-------------------------------------- Hardware Performance --------------------------------------" <<  endl;
	if (! param->pipeline) { // This loop is executed
		// layer-by-layer process
		// show the detailed hardware performance for each layer
		for (int i=0; i<netStructure.size(); i++) {
		    if ( i == 0){
			cout << "-------------------- Estimation of Layer " << i+1 << " ----------------------" << endl;
			myfile << "-------------------- Estimation of Layer " << i+1 << " ----------------------" << endl;

			ChipCalculatePerformance(inputParameter, tech, cell, i, weights[i], weights[i], inputs[i], netStructure[i][6],
						netStructure, markNM, numTileEachLayer, utilizationEachLayer, speedUpEachLayer, tileLocaEachLayer,
						numPENM, desiredPESizeNM, desiredTileSizeCM, desiredPESizeCM, CMTileheight, CMTilewidth, NMTileheight, NMTilewidth,
						&layerReadLatency, &layerReadDynamicEnergy, &tileLeakage, &layerbufferLatency, &layerbufferDynamicEnergy, &layericLatency, &layericDynamicEnergy,
						&coreLatencyADC, &coreLatencyAccum, &coreLatencyOther, &coreEnergyADC, &coreEnergyAccum, &coreEnergyOther, false, &layerclkPeriod);
			if (param->synchronous) {
				layerReadLatency *= clkPeriod;
				layerbufferLatency *= clkPeriod;
				layericLatency *= clkPeriod;
				coreLatencyADC *= clkPeriod;
				coreLatencyAccum *= clkPeriod;
				coreLatencyOther *= clkPeriod;
			}

			double numTileOtherLayer = 0;
			double layerLeakageEnergy = 0;
			for (int j=0; j<netStructure.size(); j++) {
				if (j != i) {
					numTileOtherLayer += numTileEachLayer[0][j] * numTileEachLayer[1][j];
				}
			}
			layerLeakageEnergy = numTileOtherLayer*layerReadLatency*tileLeakage;

			cout << "layer" << i+1 << "'s readLatency is: " << layerReadLatency*1e9 << "ns" << endl;
			myfile << "layer" << i+1 << "'s readLatency is: " << layerReadLatency*1e9 << "ns" << endl;
			cout << "layer" << i+1 << "'s readLatency is: " << layerReadLatency*1e9 << "ns" << endl;
			myfile << "layer" << i+1 << "'s readLatency is: " << layerReadLatency*1e9 << "ns" << endl;
			cout << "layer" << i+1 << "'s readDynamicEnergy is: " << layerReadDynamicEnergy*1e12 << "pJ" << endl;
			myfile << "layer" << i+1 << "'s readDynamicEnergy is: " << layerReadDynamicEnergy*1e12 << "pJ" << endl;
			cout << "layer" << i+1 << "'s leakagePower is: " << numTileEachLayer[0][i] * numTileEachLayer[1][i] * tileLeakage*1e6 << "uW" << endl;
			myfile << "layer" << i+1 << "'s leakagePower is: " << numTileEachLayer[0][i] * numTileEachLayer[1][i] * tileLeakage*1e6 << "uW" << endl;
			cout << "layer" << i+1 << "'s leakageEnergy is: " << layerLeakageEnergy*1e12 << "pJ" << endl;
			myfile << "layer" << i+1 << "'s leakageEnergy is: " << layerLeakageEnergy*1e12 << "pJ" << endl;
			cout << "layer" << i+1 << "'s buffer latency is: " << layerbufferLatency*1e9 << "ns" << endl;
			myfile << "layer" << i+1 << "'s buffer latency is: " << layerbufferLatency*1e9 << "ns" << endl;
			cout << "layer" << i+1 << "'s buffer readDynamicEnergy is: " << layerbufferDynamicEnergy*1e12 << "pJ" << endl;
			myfile << "layer" << i+1 << "'s buffer readDynamicEnergy is: " << layerbufferDynamicEnergy*1e12 << "pJ" << endl;
			cout << "layer" << i+1 << "'s ic latency is: " << layericLatency*1e9 << "ns" << endl;
			myfile << "layer" << i+1 << "'s ic latency is: " << layericLatency*1e9 << "ns" << endl;
			cout << "layer" << i+1 << "'s ic readDynamicEnergy is: " << layericDynamicEnergy*1e12 << "pJ" << endl;
			myfile << "layer" << i+1 << "'s ic readDynamicEnergy is: " << layericDynamicEnergy*1e12 << "pJ" << endl;

            ela_perf.push_back(layerReadLatency*1e9);
            ela_perf.push_back(layerReadDynamicEnergy*1e12);

            ela_perf.push_back(coreLatencyADC*1e9);
            ela_perf.push_back(coreLatencyAccum*1e9);
            ela_perf.push_back(coreLatencyOther*1e9);
            ela_perf.push_back(coreEnergyADC*1e12);
            ela_perf.push_back(coreEnergyAccum*1e12);
            ela_perf.push_back(coreEnergyOther*1e12);
            ela_perf.push_back(layericLatency*1e9);
            ela_perf.push_back(layericDynamicEnergy*1e12);
            ela_perf.push_back(layerbufferLatency*1e9);
            ela_perf.push_back(layerbufferDynamicEnergy*1e12);

			cout << endl;
			myfile << endl;
			cout << "************************ Breakdown of Latency and Dynamic Energy *************************" << endl;
			myfile << "************************ Breakdown of Latency and Dynamic Energy *************************" << endl;
			cout << endl;
			myfile << endl;
			cout << "----------- ADC (or S/As and precharger for SRAM) readLatency is : " << coreLatencyADC*1e9 << "ns" << endl;
			myfile << "----------- ADC (or S/As and precharger for SRAM) readLatency is : " << coreLatencyADC*1e9 << "ns" << endl;
			cout << "----------- Accumulation Circuits (subarray level: adders, shiftAdds; PE/Tile/Global level: accumulation units) readLatency is : " << coreLatencyAccum*1e9 << "ns" << endl;
			myfile << "----------- Accumulation Circuits (subarray level: adders, shiftAdds; PE/Tile/Global level: accumulation units) readLatency is : " << coreLatencyAccum*1e9 << "ns" << endl;
			cout << "----------- Other Peripheries (e.g. decoders, mux, switchmatrix, buffers, IC, pooling and activation units) readLatency is : " << coreLatencyOther*1e9 << "ns" << endl;
			myfile << "----------- Other Peripheries (e.g. decoders, mux, switchmatrix, buffers, IC, pooling and activation units) readLatency is : " << coreLatencyOther*1e9 << "ns" << endl;
			cout << "----------- ADC (or S/As and precharger for SRAM) readDynamicEnergy is : " << coreEnergyADC*1e12 << "pJ" << endl;
			myfile << "----------- ADC (or S/As and precharger for SRAM) readDynamicEnergy is : " << coreEnergyADC*1e12 << "pJ" << endl;
			cout << "----------- Accumulation Circuits (subarray level: adders, shiftAdds; PE/Tile/Global level: accumulation units) readDynamicEnergy is : " << coreEnergyAccum*1e12 << "pJ" << endl;
			myfile << "----------- Accumulation Circuits (subarray level: adders, shiftAdds; PE/Tile/Global level: accumulation units) readDynamicEnergy is : " << coreEnergyAccum*1e12 << "pJ" << endl;
			cout << "----------- Other Peripheries (e.g. decoders, mux, switchmatrix, buffers, IC, pooling and activation units) readDynamicEnergy is : " << coreEnergyOther*1e12 << "pJ" << endl;
			myfile << "----------- Other Peripheries (e.g. decoders, mux, switchmatrix, buffers, IC, pooling and activation units) readDynamicEnergy is : " << coreEnergyOther*1e12 << "pJ" << endl;
			cout << endl;
			myfile << endl;
			cout << "************************ Breakdown of Latency and Dynamic Energy *************************" << endl;
			myfile << "************************ Breakdown of Latency and Dynamic Energy *************************" << endl;
			cout << endl;
			myfile << endl;

			chipReadLatency += layerReadLatency;
			chipReadDynamicEnergy += layerReadDynamicEnergy;
			chipLeakageEnergy += layerLeakageEnergy;
			chipLeakage += tileLeakage*numTileEachLayer[0][i] * numTileEachLayer[1][i];
			chipbufferLatency += layerbufferLatency;
			chipbufferReadDynamicEnergy += layerbufferDynamicEnergy;
			chipicLatency += layericLatency;
			chipicReadDynamicEnergy += layericDynamicEnergy;

			chipLatencyADC += coreLatencyADC;
			chipLatencyAccum += coreLatencyAccum;
			chipLatencyOther += coreLatencyOther;
			chipEnergyADC += coreEnergyADC;
			chipEnergyAccum += coreEnergyAccum;
			chipEnergyOther += coreEnergyOther;
		}
		}
	} else {
		// pipeline system
		// firstly define system clock
		double systemClock = 0;

		vector<double> readLatencyPerLayer;
		vector<double> readDynamicEnergyPerLayer;
		vector<double> leakagePowerPerLayer;
		vector<double> bufferLatencyPerLayer;
		vector<double> bufferEnergyPerLayer;
		vector<double> icLatencyPerLayer;
		vector<double> icEnergyPerLayer;

		vector<double> coreLatencyADCPerLayer;
		vector<double> coreEnergyADCPerLayer;
		vector<double> coreLatencyAccumPerLayer;
		vector<double> coreEnergyAccumPerLayer;
		vector<double> coreLatencyOtherPerLayer;
		vector<double> coreEnergyOtherPerLayer;

		for (int i=0; i<netStructure.size(); i++) {
			ChipCalculatePerformance(inputParameter, tech, cell, i, weights[i], weights[i], inputs[i], netStructure[i][6],
						netStructure, markNM, numTileEachLayer, utilizationEachLayer, speedUpEachLayer, tileLocaEachLayer,
						numPENM, desiredPESizeNM, desiredTileSizeCM, desiredPESizeCM, CMTileheight, CMTilewidth, NMTileheight, NMTilewidth,
						&layerReadLatency, &layerReadDynamicEnergy, &tileLeakage, &layerbufferLatency, &layerbufferDynamicEnergy, &layericLatency, &layericDynamicEnergy,
						&coreLatencyADC, &coreLatencyAccum, &coreLatencyOther, &coreEnergyADC, &coreEnergyAccum, &coreEnergyOther, false, &layerclkPeriod);
			if (param->synchronous) {
				layerReadLatency *= clkPeriod;
				layerbufferLatency *= clkPeriod;
				layericLatency *= clkPeriod;
				coreLatencyADC *= clkPeriod;
				coreLatencyAccum *= clkPeriod;
				coreLatencyOther *= clkPeriod;
			}

			systemClock = MAX(systemClock, layerReadLatency);

			readLatencyPerLayer.push_back(layerReadLatency);
			readDynamicEnergyPerLayer.push_back(layerReadDynamicEnergy);
			leakagePowerPerLayer.push_back(numTileEachLayer[0][i] * numTileEachLayer[1][i] * tileLeakage);
			bufferLatencyPerLayer.push_back(layerbufferLatency);
			bufferEnergyPerLayer.push_back(layerbufferDynamicEnergy);
			icLatencyPerLayer.push_back(layericLatency);
			icEnergyPerLayer.push_back(layericDynamicEnergy);

			coreLatencyADCPerLayer.push_back(coreLatencyADC);
			coreEnergyADCPerLayer.push_back(coreEnergyADC);
			coreLatencyAccumPerLayer.push_back(coreLatencyAccum);
			coreEnergyAccumPerLayer.push_back(coreEnergyAccum);
			coreLatencyOtherPerLayer.push_back(coreLatencyOther);
			coreEnergyOtherPerLayer.push_back(coreEnergyOther);
		}

		for (int i=0; i<netStructure.size(); i++) {
		    if (i==0){

			cout << "-------------------- Estimation of Layer " << i+1 << " ----------------------" << endl;
			myfile << "-------------------- Estimation of Layer " << i+1 << " ----------------------" << endl;

			cout << "layer" << i+1 << "'s readLatency is: " << readLatencyPerLayer[i]*1e9 << "ns" << endl;
			myfile << "layer" << i+1 << "'s readLatency is: " << readLatencyPerLayer[i]*1e9 << "ns" << endl;
			cout << "layer" << i+1 << "'s readDynamicEnergy is: " << readDynamicEnergyPerLayer[i]*1e12 << "pJ" << endl;
			myfile << "layer" << i+1 << "'s readDynamicEnergy is: " << readDynamicEnergyPerLayer[i]*1e12 << "pJ" << endl;
			cout << "layer" << i+1 << "'s leakagePower is: " << leakagePowerPerLayer[i]*1e6 << "uW" << endl;
			myfile << "layer" << i+1 << "'s leakagePower is: " << leakagePowerPerLayer[i]*1e6 << "uW" << endl;
			cout << "layer" << i+1 << "'s leakageEnergy is: " << leakagePowerPerLayer[i] * (systemClock-readLatencyPerLayer[i]) *1e12 << "pJ" << endl;
			myfile << "layer" << i+1 << "'s leakageEnergy is: " << leakagePowerPerLayer[i] * (systemClock-readLatencyPerLayer[i]) *1e12 << "pJ" << endl;
			cout << "layer" << i+1 << "'s buffer latency is: " << bufferLatencyPerLayer[i]*1e9 << "ns" << endl;
			myfile << "layer" << i+1 << "'s buffer latency is: " << bufferLatencyPerLayer[i]*1e9 << "ns" << endl;
			cout << "layer" << i+1 << "'s buffer readDynamicEnergy is: " << bufferEnergyPerLayer[i]*1e12 << "pJ" << endl;
			myfile << "layer" << i+1 << "'s buffer readDynamicEnergy is: " << bufferEnergyPerLayer[i]*1e12 << "pJ" << endl;
			cout << "layer" << i+1 << "'s ic latency is: " << icLatencyPerLayer[i]*1e9 << "ns" << endl;
			myfile << "layer" << i+1 << "'s ic latency is: " << icLatencyPerLayer[i]*1e9 << "ns" << endl;
			cout << "layer" << i+1 << "'s ic readDynamicEnergy is: " << icEnergyPerLayer[i]*1e12 << "pJ" << endl;
			myfile << "layer" << i+1 << "'s ic readDynamicEnergy is: " << icEnergyPerLayer[i]*1e12 << "pJ" << endl;

            ela_perf.push_back(layerReadLatency*1e9);
            ela_perf.push_back(layerReadDynamicEnergy*1e12);

//            ela_perf.push_back()
//            ela_perf.push_back(coreLatencyADCPerLayer[i]*1e9)
			cout << endl;
			myfile << endl;
			cout << "************************ Breakdown of Latency and Dynamic Energy *************************" << endl;
			myfile << "************************ Breakdown of Latency and Dynamic Energy *************************" << endl;
			cout << endl;
			myfile << endl;
			cout << "----------- ADC (or S/As and precharger for SRAM) readLatency is : " << coreLatencyADCPerLayer[i]*1e9 << "ns" << endl;
			myfile << "----------- ADC (or S/As and precharger for SRAM) readLatency is : " << coreLatencyADCPerLayer[i]*1e9 << "ns" << endl;
			cout << "----------- Accumulation Circuits (subarray level: adders, shiftAdds; PE/Tile/Global level: accumulation units) readLatency is : " << coreLatencyAccumPerLayer[i]*1e9 << "ns" << endl;
			myfile << "----------- Accumulation Circuits (subarray level: adders, shiftAdds; PE/Tile/Global level: accumulation units) readLatency is : " << coreLatencyAccumPerLayer[i]*1e9 << "ns" << endl;
			cout << "----------- Other Peripheries (e.g. decoders, mux, switchmatrix, buffers, IC, pooling and activation units) readLatency is : " << coreLatencyOtherPerLayer[i]*1e9 << "ns" << endl;
			myfile << "----------- Other Peripheries (e.g. decoders, mux, switchmatrix, buffers, IC, pooling and activation units) readLatency is : " << coreLatencyOtherPerLayer[i]*1e9 << "ns" << endl;
			cout << "----------- ADC (or S/As and precharger for SRAM) readDynamicEnergy is : " << coreEnergyADCPerLayer[i]*1e12 << "pJ" << endl;
			myfile << "----------- ADC (or S/As and precharger for SRAM) readDynamicEnergy is : " << coreEnergyADCPerLayer[i]*1e12 << "pJ" << endl;
			cout << "----------- Accumulation Circuits (subarray level: adders, shiftAdds; PE/Tile/Global level: accumulation units) readDynamicEnergy is : " << coreEnergyAccumPerLayer[i]*1e12 << "pJ" << endl;
			myfile << "----------- Accumulation Circuits (subarray level: adders, shiftAdds; PE/Tile/Global level: accumulation units) readDynamicEnergy is : " << coreEnergyAccumPerLayer[i]*1e12 << "pJ" << endl;
			cout << "----------- Other Peripheries (e.g. decoders, mux, switchmatrix, buffers, IC, pooling and activation units) readDynamicEnergy is : " << coreEnergyOtherPerLayer[i]*1e12 << "pJ" << endl;
			myfile << "----------- Other Peripheries (e.g. decoders, mux, switchmatrix, buffers, IC, pooling and activation units) readDynamicEnergy is : " << coreEnergyOtherPerLayer[i]*1e12 << "pJ" << endl;
			cout << endl;
			myfile << endl;
			cout << "************************ Breakdown of Latency and Dynamic Energy *************************" << endl;
			myfile << "************************ Breakdown of Latency and Dynamic Energy *************************" << endl;
			cout << endl;
			myfile << endl;

			chipReadLatency = systemClock;
			chipReadDynamicEnergy += readDynamicEnergyPerLayer[i];
			chipLeakageEnergy += leakagePowerPerLayer[i] * (systemClock-readLatencyPerLayer[i]);
			chipLeakage += leakagePowerPerLayer[i];
			chipbufferLatency = MAX(chipbufferLatency, bufferLatencyPerLayer[i]);
			chipbufferReadDynamicEnergy += bufferEnergyPerLayer[i];
			chipicLatency = MAX(chipicLatency, icLatencyPerLayer[i]);
			chipicReadDynamicEnergy += icEnergyPerLayer[i];

			chipLatencyADC = MAX(chipLatencyADC, coreLatencyADCPerLayer[i]);
			chipLatencyAccum = MAX(chipLatencyAccum, coreLatencyAccumPerLayer[i]);
			chipLatencyOther = MAX(chipLatencyOther, coreLatencyOtherPerLayer[i]);
			chipEnergyADC += coreEnergyADCPerLayer[i];
			chipEnergyAccum += coreEnergyAccumPerLayer[i];
			chipEnergyOther += coreEnergyOtherPerLayer[i];
		}
        }
	}

	cout << "------------------------------ Summary --------------------------------" <<  endl;
	myfile << "------------------------------ Summary --------------------------------" <<  endl;
	cout << endl;
	myfile << endl;
	cout << "ChipArea : " << chipArea*1e12 << "um^2" << endl;
	myfile << "ChipArea : " << chipArea*1e12 << "um^2" << endl;
	cout << "Chip total CIM array : " << chipAreaArray*1e12 << "um^2" << endl;
	myfile << "Chip total CIM array : " << chipAreaArray*1e12 << "um^2" << endl;
	cout << "Total IC Area on chip (Global and Tile/PE local): " << chipAreaIC*1e12 << "um^2" << endl;
	myfile << "Total IC Area on chip (Global and Tile/PE local): " << chipAreaIC*1e12 << "um^2" << endl;
	cout << "Total ADC (or S/As and precharger for SRAM) Area on chip : " << chipAreaADC*1e12 << "um^2" << endl;
	myfile << "Total ADC (or S/As and precharger for SRAM) Area on chip : " << chipAreaADC*1e12 << "um^2" << endl;
	cout << "Total Accumulation Circuits (subarray level: adders, shiftAdds; PE/Tile/Global level: accumulation units) on chip : " << chipAreaAccum*1e12 << "um^2" << endl;
	myfile << "Total Accumulation Circuits (subarray level: adders, shiftAdds; PE/Tile/Global level: accumulation units) on chip : " << chipAreaAccum*1e12 << "um^2" << endl;
	cout << "Other Peripheries (e.g. decoders, mux, switchmatrix, buffers, pooling and activation units) : " << chipAreaOther*1e12 << "um^2" << endl;
	myfile << "Other Peripheries (e.g. decoders, mux, switchmatrix, buffers, pooling and activation units) : " << chipAreaOther*1e12 << "um^2" << endl;
	cout << endl;
	myfile << endl;
	if (! param->pipeline) {  //This is used
		if (param->synchronous) {
		cout << "Chip clock period is: " << clkPeriod*1e9 << "ns" <<endl;
		myfile << "Chip clock period is: " << clkPeriod*1e9 << "ns" <<endl;
		}
		cout << "Chip layer-by-layer readLatency (per image) is: " << chipReadLatency*1e9 << "ns" << endl;
		myfile << "Chip layer-by-layer readLatency (per image) is: " << chipReadLatency*1e9 << "ns" << endl;
		cout << "Chip total readDynamicEnergy is: " << chipReadDynamicEnergy*1e12 << "pJ" << endl;
		myfile << "Chip total readDynamicEnergy is: " << chipReadDynamicEnergy*1e12 << "pJ" << endl;
		cout << "Chip total leakage Energy is: " << chipLeakageEnergy*1e12 << "pJ" << endl;
		myfile << "Chip total leakage Energy is: " << chipLeakageEnergy*1e12 << "pJ" << endl;
		cout << "Chip total leakage Power is: " << chipLeakage*1e6 << "uW" << endl;
		myfile << "Chip total leakage Power is: " << chipLeakage*1e6 << "uW" << endl;
		cout << "Chip buffer readLatency is: " << chipbufferLatency*1e9 << "ns" << endl;
		myfile << "Chip buffer readLatency is: " << chipbufferLatency*1e9 << "ns" << endl;
		cout << "Chip buffer readDynamicEnergy is: " << chipbufferReadDynamicEnergy*1e12 << "pJ" << endl;
		myfile << "Chip buffer readDynamicEnergy is: " << chipbufferReadDynamicEnergy*1e12 << "pJ" << endl;
		cout << "Chip ic readLatency is: " << chipicLatency*1e9 << "ns" << endl;
		myfile << "Chip ic readLatency is: " << chipicLatency*1e9 << "ns" << endl;
		cout << "Chip ic readDynamicEnergy is: " << chipicReadDynamicEnergy*1e12 << "pJ" << endl;
		myfile << "Chip ic readDynamicEnergy is: " << chipicReadDynamicEnergy*1e12 << "pJ" << endl;
	} else {
		if (param->synchronous) {
		cout << "Chip clock period is: " << clkPeriod*1e9 << "ns" <<endl;
		myfile << "Chip clock period is: " << clkPeriod*1e9 << "ns" <<endl;
		}
		cout << "Chip pipeline-system-clock-cycle (per image) is: " << chipReadLatency*1e9 << "ns" << endl;
		myfile << "Chip pipeline-system-clock-cycle (per image) is: " << chipReadLatency*1e9 << "ns" << endl;
		cout << "Chip pipeline-system readDynamicEnergy (per image) is: " << chipReadDynamicEnergy*1e12 << "pJ" << endl;
		myfile << "Chip pipeline-system readDynamicEnergy (per image) is: " << chipReadDynamicEnergy*1e12 << "pJ" << endl;
		cout << "Chip pipeline-system leakage Energy (per image) is: " << chipLeakageEnergy*1e12 << "pJ" << endl;
		myfile << "Chip pipeline-system leakage Energy (per image) is: " << chipLeakageEnergy*1e12 << "pJ" << endl;
		cout << "Chip pipeline-system leakage Power (per image) is: " << chipLeakage*1e6 << "uW" << endl;
		myfile << "Chip pipeline-system leakage Power (per image) is: " << chipLeakage*1e6 << "uW" << endl;
		cout << "Chip pipeline-system buffer readLatency (per image) is: " << chipbufferLatency*1e9 << "ns" << endl;
		myfile << "Chip pipeline-system buffer readLatency (per image) is: " << chipbufferLatency*1e9 << "ns" << endl;
		cout << "Chip pipeline-system buffer readDynamicEnergy (per image) is: " << chipbufferReadDynamicEnergy*1e12 << "pJ" << endl;
		myfile << "Chip pipeline-system buffer readDynamicEnergy (per image) is: " << chipbufferReadDynamicEnergy*1e12 << "pJ" << endl;
		cout << "Chip pipeline-system ic readLatency (per image) is: " << chipicLatency*1e9 << "ns" << endl;
		myfile << "Chip pipeline-system ic readLatency (per image) is: " << chipicLatency*1e9 << "ns" << endl;
		cout << "Chip pipeline-system ic readDynamicEnergy (per image) is: " << chipicReadDynamicEnergy*1e12 << "pJ" << endl;
		myfile << "Chip pipeline-system ic readDynamicEnergy (per image) is: " << chipicReadDynamicEnergy*1e12 << "pJ" << endl;
	}

	cout << endl;
	myfile << endl;
	cout << "************************ Breakdown of Latency and Dynamic Energy *************************" << endl;
	myfile << "************************ Breakdown of Latency and Dynamic Energy *************************" << endl;
	cout << endl;
	myfile << endl;
	cout << "----------- ADC (or S/As and precharger for SRAM) readLatency is : " << chipLatencyADC*1e9 << "ns" << endl;
	myfile << "----------- ADC (or S/As and precharger for SRAM) readLatency is : " << chipLatencyADC*1e9 << "ns" << endl;
	cout << "----------- Accumulation Circuits (subarray level: adders, shiftAdds; PE/Tile/Global level: accumulation units) readLatency is : " << chipLatencyAccum*1e9 << "ns" << endl;
	myfile << "----------- Accumulation Circuits (subarray level: adders, shiftAdds; PE/Tile/Global level: accumulation units) readLatency is : " << chipLatencyAccum*1e9 << "ns" << endl;
	cout << "----------- Other Peripheries (e.g. decoders, mux, switchmatrix, buffers, IC, pooling and activation units) readLatency is : " << chipLatencyOther*1e9 << "ns" << endl;
	myfile << "----------- Other Peripheries (e.g. decoders, mux, switchmatrix, buffers, IC, pooling and activation units) readLatency is : " << chipLatencyOther*1e9 << "ns" << endl;
	cout << "----------- ADC (or S/As and precharger for SRAM) readDynamicEnergy is : " << chipEnergyADC*1e12 << "pJ" << endl;
	myfile << "----------- ADC (or S/As and precharger for SRAM) readDynamicEnergy is : " << chipEnergyADC*1e12 << "pJ" << endl;
	cout << "----------- Accumulation Circuits (subarray level: adders, shiftAdds; PE/Tile/Global level: accumulation units) readDynamicEnergy is : " << chipEnergyAccum*1e12 << "pJ" << endl;
	myfile << "----------- Accumulation Circuits (subarray level: adders, shiftAdds; PE/Tile/Global level: accumulation units) readDynamicEnergy is : " << chipEnergyAccum*1e12 << "pJ" << endl;
	cout << "----------- Other Peripheries (e.g. decoders, mux, switchmatrix, buffers, IC, pooling and activation units) readDynamicEnergy is : " << chipEnergyOther*1e12 << "pJ" << endl;
	myfile << "----------- Other Peripheries (e.g. decoders, mux, switchmatrix, buffers, IC, pooling and activation units) readDynamicEnergy is : " << chipEnergyOther*1e12 << "pJ" << endl;
	cout << endl;
	myfile << endl;
	cout << "************************ Breakdown of Latency and Dynamic Energy *************************" << endl;
	myfile << "************************ Breakdown of Latency and Dynamic Energy *************************" << endl;
	cout << endl;
	myfile << endl;

	cout << endl;
	myfile << endl;
	cout << "----------------------------- Performance -------------------------------" << endl;
	myfile << "----------------------------- Performance -------------------------------" << endl;
	if (! param->pipeline) {
		if(param->validated){
			cout << "Energy Efficiency TOPS/W (Layer-by-Layer Process): " << numComputation/(chipReadDynamicEnergy*1e12+chipLeakageEnergy*1e12)/param->zeta << endl;	// post-layout energy increase, zeta = 1.23 by default
			myfile << "Energy Efficiency TOPS/W (Layer-by-Layer Process): " << numComputation/(chipReadDynamicEnergy*1e12+chipLeakageEnergy*1e12)/param->zeta << endl;
			cout << "Num Computations: " <<  numComputation << endl;
			myfile << "Num Computations: " <<  numComputation << endl; // post-layout energy increase, zeta = 1.23 by default
		}else{
			cout << "Energy Efficiency TOPS/W (Layer-by-Layer Process): " << numComputation/(chipReadDynamicEnergy*1e12+chipLeakageEnergy*1e12) << endl;
			myfile << "Energy Efficiency TOPS/W (Layer-by-Layer Process): " << numComputation/(chipReadDynamicEnergy*1e12+chipLeakageEnergy*1e12) << endl;
		}
		cout << "Throughput TOPS (Layer-by-Layer Process): " << numComputation/(chipReadLatency*1e12) << endl;
		myfile << "Throughput TOPS (Layer-by-Layer Process): " << numComputation/(chipReadLatency*1e12) << endl;
		cout << "Throughput FPS (Layer-by-Layer Process): " << 1/(chipReadLatency) << endl;
		myfile << "Throughput FPS (Layer-by-Layer Process): " << 1/(chipReadLatency) << endl;
		cout << "Compute efficiency TOPS/mm^2 (Layer-by-Layer Process): " << numComputation/(chipReadLatency*1e12)/(chipArea*1e6) << endl;
		myfile << "Compute efficiency TOPS/mm^2 (Layer-by-Layer Process): " << numComputation/(chipReadLatency*1e12)/(chipArea*1e6) << endl;
	} else {
		if(param->validated){
			cout << "Energy Efficiency TOPS/W (Pipelined Process): " << numComputation/(chipReadDynamicEnergy*1e12+chipLeakageEnergy*1e12)/param->zeta << endl;	// post-layout energy increase, zeta = 1.23 by default
			myfile << "Energy Efficiency TOPS/W (Pipelined Process): " << numComputation/(chipReadDynamicEnergy*1e12+chipLeakageEnergy*1e12)/param->zeta << endl;	// post-layout energy increase, zeta = 1.23 by default
		}else{
			cout << "Energy Efficiency TOPS/W (Pipelined Process): " << numComputation/(chipReadDynamicEnergy*1e12+chipLeakageEnergy*1e12) << endl;
			myfile << "Energy Efficiency TOPS/W (Pipelined Process): " << numComputation/(chipReadDynamicEnergy*1e12+chipLeakageEnergy*1e12) << endl;
		}
		cout << "Throughput TOPS (Pipelined Process): " << numComputation/(chipReadLatency*1e12) << endl;
		myfile << "Throughput TOPS (Pipelined Process): " << numComputation/(chipReadLatency*1e12) << endl;
		cout << "Throughput FPS (Pipelined Process): " << 1/(chipReadLatency) << endl;
		myfile << "Throughput FPS (Pipelined Process): " << 1/(chipReadLatency) << endl;
		cout << "Compute efficiency TOPS/mm^2 (Pipelined Process): " << numComputation/(chipReadLatency*1e12)/(chipArea*1e6) << endl;
		myfile << "Compute efficiency TOPS/mm^2 (Pipelined Process): " << numComputation/(chipReadLatency*1e12)/(chipArea*1e6) << endl;
	}
	cout << "-------------------------------------- Hardware Performance Done --------------------------------------" <<  endl;
	myfile << "-------------------------------------- Hardware Performance Done --------------------------------------" <<  endl;
	cout << endl;
	myfile << endl;
	auto stop = chrono::high_resolution_clock::now();
	auto duration = chrono::duration_cast<chrono::seconds>(stop-start);
    cout << "------------------------------ Simulation Performance --------------------------------" <<  endl;
    myfile << "------------------------------ Simulation Performance --------------------------------" <<  endl;
	cout << "Total Run-time of NeuroSim: " << duration.count() << " seconds" << endl;
	myfile << "Total Run-time of NeuroSim: " << duration.count() << " seconds" << endl;
	cout << "------------------------------ Simulation Performance --------------------------------" <<  endl;
	myfile << "------------------------------ Simulation Performance --------------------------------" <<  endl;
    myfile.close();

	return ela_perf;
}

int main(int argc, char * argv[]){

    auto start = chrono::high_resolution_clock::now();

	gen.seed(0);

	vector<vector<double> > netStructure;
	netStructure = getNetStructure(argv[1]);
    double latency = 0;
    double energy = 0;
    double computations = 0, ADC_latency=0, Acc_latency=0, Other_latency=0, ADC_energy=0, Acc_energy=0, Other_energy=0, ic_latency=0, ic_energy=0, buff_latency = 0, buff_energy = 0;
    for (int i =0 ; i < 12; i++){


        cout << "LAYER BEING EVALUATED " << i+1 << endl;
        vector<vector<double>> mini_netStructure;
        int val;
        vector<string> weights;
        vector<string> inputs;

//        weights.push_back(argv[4*(i+1)]);
//        weights.push_back(argv[4*(i+1)+2]);

//        inputs.push_back(argv[5+i*4]);
//        inputs.push_back(argv[7+i*4]);

//        mini_netStructure.push_back(netStructure[i*2]);
//        mini_netStructure.push_back(netStructure[i*2+1]);
        weights.push_back(argv[4*i+4]);
        weights.push_back(argv[4*i+6]);
//        weights.push_back(argv[6*i+8]);

        inputs.push_back(argv[4*i+5]);
        inputs.push_back(argv[4*i+7]);
//        inputs.push_back(argv[6*i+9]);

        mini_netStructure.push_back(netStructure[2*i]);
        mini_netStructure.push_back(netStructure[2*i+1]);
//        mini_netStructure.push_back(netStructure[3*i+2]);


        vector<double> ela_perf = ela_calc(mini_netStructure, weights, inputs, i);

        cout << "############  LAYER " << i+1 << "   ##############"<< endl;

        cout << "Computations  " << ela_perf[0] << endl;
        cout << "Latency  " << ela_perf[1] << endl;
        cout << "Energy  " << ela_perf[2] << endl;

        latency = latency + ela_perf[1];
        energy = energy + ela_perf[2];
        computations = computations + ela_perf[0];

        ADC_latency += ela_perf[3];
        Acc_latency += ela_perf[4];
        Other_latency += ela_perf[5];

        ADC_energy += ela_perf[6];
        Acc_energy += ela_perf[7];
        Other_energy += ela_perf[8];

        ic_latency += ela_perf[9];
        ic_energy += ela_perf[10];

        buff_latency += ela_perf[11];
        buff_energy += ela_perf[12];

	}

	cout << "{'Computations': " << computations << "," <<  endl;
	cout << "'Energy': " << energy << "," << endl;
	cout << "'Latency': " << latency << "," << endl;
	cout << "'TOPS/Watt': " << computations/(energy)/param->zeta << "," << endl;

    cout << "'L_ADC': " << ADC_latency << "," << endl;
    cout << "'L_Acc': " << Acc_latency << "," << endl;
    cout << "'L_Other': " << Other_latency << "," << endl;

    cout << "'E_ADC': " << ADC_energy << "," << endl;
    cout << "'E_Acc': " << Acc_energy << "," << endl;
    cout << "'E_Other': " << Other_energy << "," << endl;

    cout << "'L_IC': " << ic_latency << "," << endl;
    cout << "'E_IC': " << ic_energy << "," << endl;

    cout << "'L_Bu' " << buff_latency << "," << endl;
    cout << "'E_Bu' " << buff_energy << "}" << endl;
//	cout << "TOPS/mm2 " << computations/(latency)/(chipArea*1e6)endl;
//	cout << "EDP " << energy*latency << "ns*pJ" << endl;
//	cout << "TOPS " << endl;


    return 0;
}
//int main(int argc, char * argv[]) {
//
//	auto start = chrono::high_resolution_clock::now();
//
//	gen.seed(0);
//
//	vector<vector<double> > netStructure;
//	netStructure = getNetStructure(argv[1]);
//
//	// define weight/input/memory precision from wrapper
//	param->synapseBit = atoi(argv[2]);              // precision of synapse weight
//	param->numBitInput = atoi(argv[3]);             // precision of input neural activation
//    param->numColMuxed = atoi(argv[4]);
//
//    if (atoi(argv[5]) == 0){
//        param->SARADC = false;
//    }
//    else
//        param->SARADC = true;
//
//
//
//    cout << "SARADC   "<< param->SARADC << endl;
////	param->numBitInput = 8;
//
//	if (param->cellBit > param->synapseBit) {
//		cout << "ERROR!: Memory precision is even higher than synapse precision, please modify 'cellBit' in Param.cpp!" << endl;
//		param->cellBit = param->synapseBit;
//	}
//
//	/*** initialize operationMode as default ***/
//	param->conventionalParallel = 0;
//	param->conventionalSequential = 0;
//	param->BNNparallelMode = 0;                // parallel BNN
//	param->BNNsequentialMode = 0;              // sequential BNN
//	param->XNORsequentialMode = 0;           // Use several multi-bit RRAM as one synapse
//	param->XNORparallelMode = 0;         // Use several multi-bit RRAM as one synapse
//	switch(param->operationmode) {
//		case 6:	    param->XNORparallelMode = 1;               break;
//		case 5:	    param->XNORsequentialMode = 1;             break;
//		case 4:	    param->BNNparallelMode = 1;                break;
//		case 3:	    param->BNNsequentialMode = 1;              break;
//		case 2:	    param->conventionalParallel = 1;           break;
//		case 1:	    param->conventionalSequential = 1;         break;
//		case -1:	break;
//		default:	exit(-1);
//	}
//
//	if (param->XNORparallelMode || param->XNORsequentialMode) {
//		param->numRowPerSynapse = 2;
//	} else {
//		param->numRowPerSynapse = 1;
//	}
//	if (param->BNNparallelMode) {
//		param->numColPerSynapse = 2;
//	} else if (param->XNORparallelMode || param->XNORsequentialMode || param->BNNsequentialMode) {
//		param->numColPerSynapse = 1;
//	} else {
//		param->numColPerSynapse = ceil((double)param->synapseBit/(double)param->cellBit);
//	}
//	for (int i =0 ; i < netStructure.size(); i++){
//	    for (int j = 0; j < netStructure[i].size(); j++){
//	        cout << netStructure[i][j] << ",";
//	    }
//	    cout << endl;
//	}
//	double maxPESizeNM, maxTileSizeCM, numPENM;
//	vector<int> markNM;
//	vector<int> pipelineSpeedUp;
//	markNM = ChipDesignInitialize(inputParameter, tech, cell, false, netStructure, &maxPESizeNM, &maxTileSizeCM, &numPENM);
//	pipelineSpeedUp = ChipDesignInitialize(inputParameter, tech, cell, true, netStructure, &maxPESizeNM, &maxTileSizeCM, &numPENM);
//
//	double desiredNumTileNM, desiredPESizeNM, desiredNumTileCM, desiredTileSizeCM, desiredPESizeCM;
//	int numTileRow, numTileCol;
//
//	vector<vector<double> > numTileEachLayer;
//	vector<vector<double> > utilizationEachLayer;
//	vector<vector<double> > speedUpEachLayer;
//	vector<vector<double> > tileLocaEachLayer;
//
//	numTileEachLayer = ChipFloorPlan(true, false, false, netStructure, markNM,
//					maxPESizeNM, maxTileSizeCM, numPENM, pipelineSpeedUp,
//					&desiredNumTileNM, &desiredPESizeNM, &desiredNumTileCM, &desiredTileSizeCM, &desiredPESizeCM, &numTileRow, &numTileCol);
////    numPENM = 9;
////    desiredTileSizeCM = 128;
////    desiredPESizeCM = 128;
////    desiredPESizeNM = 256;
////    numTileEachLayer = {{1,1,1,1,1,1,1,1,1,1,1,1}, {2.0, 3.0, 72.0, 4.0, 25.0, 13.0, 13.0, 7.0, 18.0, 50.0, 7.0, 13.0}};
////    numTileEachLayer = {{1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1}, {2.0, 25.0, 7.0, 1.0, 7.0, 7.0, 14.0, 26.0, 2.0, 26.0, 26.0, 13.0, 7.0, 1.0, 7.0, 7.0, 14.0, 26.0, 2.0, 26.0, 26.0}};
////    for (int i = 0; i < numTileEachLayer[0].size(); i++) {
////           numTileEachLayer[0][i] = 1;
////           numTileEachLayer[1][i] = 4;
////        }
////	cout << "desiredPESizeNM " << desiredPESizeNM << endl;
//	utilizationEachLayer = ChipFloorPlan(false, true, false, netStructure, markNM,
//					maxPESizeNM, maxTileSizeCM, numPENM, pipelineSpeedUp,
//					&desiredNumTileNM, &desiredPESizeNM, &desiredNumTileCM, &desiredTileSizeCM, &desiredPESizeCM, &numTileRow, &numTileCol);
////	numPENM = 9;
////	desiredTileSizeCM = 128;
////	desiredPESizeCM = 128;
////    desiredPESizeNM = 256;
//	speedUpEachLayer = ChipFloorPlan(false, false, true, netStructure, markNM,
//					maxPESizeNM, maxTileSizeCM, numPENM, pipelineSpeedUp,
//					&desiredNumTileNM, &desiredPESizeNM, &desiredNumTileCM, &desiredTileSizeCM, &desiredPESizeCM, &numTileRow, &numTileCol);
////    numPENM = 9;
////    desiredTileSizeCM = 128;
////    desiredPESizeCM = 128;
////    desiredPESizeNM = 256;
////    speedUpEachLayer = {{1,1,1,1,1,1,1,1,1,1,1,1}, {8.0, 1.0, 16.0, 1.0, 16.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0}};
////    for (int i = 0; i < speedUpEachLayer[0].size(); i++) {
////           speedUpEachLayer[0][i] = 4;
////           speedUpEachLayer[1][i] = 1;
////        }
////    speedUpEachLayer =
//	tileLocaEachLayer = ChipFloorPlan(false, false, false, netStructure, markNM,
//					maxPESizeNM, maxTileSizeCM, numPENM, pipelineSpeedUp,
//					&desiredNumTileNM, &desiredPESizeNM, &desiredNumTileCM, &desiredTileSizeCM, &desiredPESizeCM, &numTileRow, &numTileCol);
////    desiredPESizeNM = 512;
////    numPENM = 9;
////    desiredTileSizeCM = 128;
////    desiredPESizeCM = 128;
////    desiredPESizeNM = 256;
//	for (int i = 0; i < tileLocaEachLayer[0].size(); i++) {
//            cout << "num_tile" << tileLocaEachLayer[0][i] << " ";
//        }
//    cout << endl;
//    for (int i = 0; i < tileLocaEachLayer[1].size(); i++) {
//            cout << "num_tile" << tileLocaEachLayer[1][i] << " ";
//        }
//    ofstream myfile;
//    myfile.open ("./example.txt");
//    cout << endl;
//	cout << "------------------------------ FloorPlan --------------------------------" <<  endl;
//	myfile << "------------------------------ FloorPlan --------------------------------" <<  endl;
//	cout << endl;
//	myfile << endl;
//	cout << "Tile and PE size are optimized to maximize memory utilization ( = memory mapped by synapse / total memory on chip)" << endl;
//	myfile << "Tile and PE size are optimized to maximize memory utilization ( = memory mapped by synapse / total memory on chip)" << endl;
//	cout << endl;
//	myfile << endl;
//	if (!param->novelMapping) {
//		cout << "Desired Conventional Mapped Tile Storage Size: " << desiredTileSizeCM << "x" << desiredTileSizeCM << endl;
//        myfile << "Desired Conventional Mapped Tile Storage Size: " << desiredTileSizeCM << "x" << desiredTileSizeCM << endl;
//
//		cout << "Desired Conventional PE Storage Size: " << desiredPESizeCM << "x" << desiredPESizeCM << endl;
//        myfile << "Desired Conventional PE Storage Size: " << desiredPESizeCM << "x" << desiredPESizeCM << endl;
//
//	} else {
//		cout << "Desired Conventional Mapped Tile Storage Size: " << desiredTileSizeCM << "x" << desiredTileSizeCM << endl;
//		myfile << "Desired Conventional Mapped Tile Storage Size: " << desiredTileSizeCM << "x" << desiredTileSizeCM << endl;
//		cout << "Desired Conventional PE Storage Size: " << desiredPESizeCM << "x" << desiredPESizeCM << endl;
//		myfile << "Desired Conventional PE Storage Size: " << desiredPESizeCM << "x" << desiredPESizeCM << endl;
//		cout << "Desired Novel Mapped Tile Storage Size: " << numPENM << "x" << desiredPESizeNM << "x" << desiredPESizeNM << endl;
//		myfile << "Desired Novel Mapped Tile Storage Size: " << numPENM << "x" << desiredPESizeNM << "x" << desiredPESizeNM << endl;
//	}
//	cout << "User-defined SubArray Size: " << param->numRowSubArray << "x" << param->numColSubArray << endl;
//	myfile << "User-defined SubArray Size: " << param->numRowSubArray << "x" << param->numColSubArray << endl;
//	cout << endl;
//	myfile << endl;
//	cout << "----------------- # of tile used for each layer -----------------" <<  endl;
//	myfile << "----------------- # of tile used for each layer -----------------" <<  endl;
//	double totalNumTile = 0;
//	for (int i=0; i<netStructure.size(); i++) {
//		cout << "layer" << i+1 << ": " << numTileEachLayer[0][i] * numTileEachLayer[1][i] << endl;
//		myfile << "layer" << i+1 << ": " << numTileEachLayer[0][i] * numTileEachLayer[1][i] << endl;
//		totalNumTile += numTileEachLayer[0][i] * numTileEachLayer[1][i];
//	}
//	cout << endl;
//	myfile << endl;
//
//	cout << "----------------- Speed-up of each layer ------------------" <<  endl;
//	myfile << "----------------- Speed-up of each layer ------------------" <<  endl;
//	for (int i=0; i<netStructure.size(); i++) {
//		cout << "layer" << i+1 << ": " << speedUpEachLayer[0][i] * speedUpEachLayer[1][i] << endl;
//		myfile << "layer" << i+1 << ": " << speedUpEachLayer[0][i] * speedUpEachLayer[1][i] << endl;
//	}
//	cout << endl;
//	myfile << endl;
//
//	cout << "----------------- Utilization of each layer ------------------" <<  endl;
//	myfile << "----------------- Utilization of each layer ------------------" <<  endl;
//	double realMappedMemory = 0;
//	for (int i=0; i<netStructure.size(); i++) {
//		cout << "layer" << i+1 << ": " << utilizationEachLayer[i][0] << endl;
//		myfile << "layer" << i+1 << ": " << utilizationEachLayer[i][0] << endl;
//		realMappedMemory += numTileEachLayer[0][i] * numTileEachLayer[1][i] * utilizationEachLayer[i][0];
//	}
//	cout << "Memory Utilization of Whole Chip: " << realMappedMemory/totalNumTile*100 << " % " << endl;
//	myfile << "Memory Utilization of Whole Chip: " << realMappedMemory/totalNumTile*100 << " % " << endl;
//	cout << endl;
//	myfile << endl;
//	cout << "---------------------------- FloorPlan Done ------------------------------" <<  endl;
//	myfile << "---------------------------- FloorPlan Done ------------------------------" <<  endl;
//	cout << endl;
//	myfile << endl;
//	cout << endl;
//	myfile << endl;
//	cout << endl;
//	myfile << endl;
//
//	double numComputation = 0;
//	for (int i=0; i<netStructure.size(); i++) {
//		numComputation += 2*(netStructure[i][0] * netStructure[i][1] * netStructure[i][2] * netStructure[i][3] * netStructure[i][4] * netStructure[i][5]);
//	}
//
//	ChipInitialize(inputParameter, tech, cell, netStructure, markNM, numTileEachLayer,
//					numPENM, desiredNumTileNM, desiredPESizeNM, desiredNumTileCM, desiredTileSizeCM, desiredPESizeCM, numTileRow, numTileCol);
//
//	double chipHeight, chipWidth, chipArea, chipAreaIC, chipAreaADC, chipAreaAccum, chipAreaOther, chipAreaArray;
//	double CMTileheight = 0;
//	double CMTilewidth = 0;
//	double NMTileheight = 0;
//	double NMTilewidth = 0;
//	vector<double> chipAreaResults;
//
//	chipAreaResults = ChipCalculateArea(inputParameter, tech, cell, desiredNumTileNM, numPENM, desiredPESizeNM, desiredNumTileCM, desiredTileSizeCM, desiredPESizeCM, numTileRow,
//					&chipHeight, &chipWidth, &CMTileheight, &CMTilewidth, &NMTileheight, &NMTilewidth);
//	chipArea = chipAreaResults[0];
//	chipAreaIC = chipAreaResults[1];
//	chipAreaADC = chipAreaResults[2];
//	chipAreaAccum = chipAreaResults[3];
//	chipAreaOther = chipAreaResults[4];
//	chipAreaArray = chipAreaResults[5];
//
//	double clkPeriod = 0;
//	double layerclkPeriod = 0;
//
//	double chipReadLatency = 0;
//	double chipReadDynamicEnergy = 0;
//	double chipLeakageEnergy = 0;
//	double chipLeakage = 0;
//	double chipbufferLatency = 0;
//	double chipbufferReadDynamicEnergy = 0;
//	double chipicLatency = 0;
//	double chipicReadDynamicEnergy = 0;
//
//	double chipLatencyADC = 0;
//	double chipLatencyAccum = 0;
//	double chipLatencyOther = 0;
//	double chipEnergyADC = 0;
//	double chipEnergyAccum = 0;
//	double chipEnergyOther = 0;
//
//	double layerReadLatency = 0;
//	double layerReadDynamicEnergy = 0;
//	double tileLeakage = 0;
//	double layerbufferLatency = 0;
//	double layerbufferDynamicEnergy = 0;
//	double layericLatency = 0;
//	double layericDynamicEnergy = 0;
//
//	double coreLatencyADC = 0;
//	double coreLatencyAccum = 0;
//	double coreLatencyOther = 0;
//	double coreEnergyADC = 0;
//	double coreEnergyAccum = 0;
//	double coreEnergyOther = 0;
//
////	vector<int> wl_bit = {4, 5, 3, 5, 2, 3, 2, 3, 3, 4, 4, 2, 8, 8};
////	vector<int> adc_bit = {32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32};
//
////        [7,7,7,3,6,1,3,3]
////    param->numBitInput = 2;
//	if (param->synchronous){
//		// calculate clkFreq
//		for (int i=0; i<netStructure.size(); i++) {
////            param->numBitInput = wl_bit[i];
////            param->levelOutput = adc_bit[i];
//		    if (i == 0){
//		        cout << "i" << i << endl;
//		        param->numBitInput = 2;
//		        param->SARADC = true;
//		        param->numColMuxed = 4;
//		        }
//            if (i == 1){
//                cout << "i" << i << endl;
//		        param->numBitInput = 2;
//		        }
//            if (i == 2){
//                cout << "i" << i << endl;
//		        param->numBitInput = 2;
//		        }
//
//            if (i == 3){
//                cout << "i" << i << endl;
//		        param->numBitInput = 2;
//		        }
//
//            if (i == 4){
//                cout << "i" << i << endl;
//		        param->numBitInput = 2;
//		        }
//
//            if (i == 5){
//                cout << "i" << i << endl;
//		        param->numBitInput = 2;
//		        }
//
//            if (i == 6){
//                cout << "i" << i << endl;
//		        param->numBitInput = 2;
//		        }
//
//            if (i == 7){
//                cout << "i" << i << endl;
//		        param->numBitInput = 2;
//		        }
//
//            if (i == 8){
//                cout << "i" << i << endl;
//		        param->numBitInput = 2;
//		        }
//
//            if (i == 9){
//                cout << "i" << i << endl;
//		        param->numBitInput = 2;
//		        }
//            if (i == 10){
//                cout << "i" << i << endl;
//		        param->numBitInput = 2;
//		        }
//            if (i == 11){
//                cout << "i" << i << endl;
//		        param->numBitInput = 2;
//		        }
//
//
//
//		    cout << "synch true new" << endl;
//			ChipCalculatePerformance(inputParameter, tech, cell, i, argv[2*i+6], argv[2*i+6], argv[2*i+7], netStructure[i][6],
//						netStructure, markNM, numTileEachLayer, utilizationEachLayer, speedUpEachLayer, tileLocaEachLayer,
//						numPENM, desiredPESizeNM, desiredTileSizeCM, desiredPESizeCM, CMTileheight, CMTilewidth, NMTileheight, NMTilewidth,
//						&layerReadLatency, &layerReadDynamicEnergy, &tileLeakage, &layerbufferLatency, &layerbufferDynamicEnergy, &layericLatency, &layericDynamicEnergy,
//						&coreLatencyADC, &coreLatencyAccum, &coreLatencyOther, &coreEnergyADC, &coreEnergyAccum, &coreEnergyOther, true, &layerclkPeriod);
//			if(clkPeriod < layerclkPeriod){
//				clkPeriod = layerclkPeriod;
//			}
//		}
//		if(param->clkFreq > 1/clkPeriod){
//			param->clkFreq = 1/clkPeriod;
//		}
//	}
//
//	cout << "-------------------------------------- Hardware Performance --------------------------------------" <<  endl;
//	myfile << "-------------------------------------- Hardware Performance --------------------------------------" <<  endl;
//	if (! param->pipeline) { // This loop is executed
//		// layer-by-layer process
//		// show the detailed hardware performance for each layer
//		for (int i=0; i<netStructure.size(); i++) {
//			cout << "-------------------- Estimation of Layer " << i+1 << " ----------------------" << endl;
//			myfile << "-------------------- Estimation of Layer " << i+1 << " ----------------------" << endl;
////            param->numBitInput = wl_bit[i];
////            param->levelOutput = adc_bit[i];
//            if (i == 0){
//		        cout << "i" << i << endl;
//		        param->numBitInput = 8;
//		        param->SARADC = true;
//		        param->numColMuxed = 4;
//
//		        }
//            if (i == 1){
//                cout << "i" << i << endl;
//		        param->numBitInput = 2;
//		        }
//            if (i == 2){
//                cout << "i" << i << endl;
//		        param->numBitInput = 2;
//		        }
//
//            if (i == 3){
//                cout << "i" << i << endl;
//		        param->numBitInput = 2;
//		        }
//
//            if (i == 4){
//                cout << "i" << i << endl;
//		        param->numBitInput = 2;
//		        }
//
//            if (i == 5){
//                cout << "i" << i << endl;
//		        param->numBitInput = 2;
//		        }
//
//            if (i == 6){
//                cout << "i" << i << endl;
//		        param->numBitInput = 2;
//		        }
//
//            if (i == 7){
//                cout << "i" << i << endl;
//		        param->numBitInput = 2;
//		        }
//
//            if (i == 8){
//                cout << "i" << i << endl;
//		        param->numBitInput = 2;
//		        }
//
//            if (i == 9){
//                cout << "i" << i << endl;
//		        param->numBitInput = 2;
//		        }
//            if (i == 10){
//                cout << "i" << i << endl;
//		        param->numBitInput = 2;
//		        }
//            if (i == 11){
//                cout << "i" << i << endl;
//		        param->numBitInput = 2;
//		        }
//			ChipCalculatePerformance(inputParameter, tech, cell, i, argv[2*i+6], argv[2*i+6], argv[2*i+7], netStructure[i][6],
//						netStructure, markNM, numTileEachLayer, utilizationEachLayer, speedUpEachLayer, tileLocaEachLayer,
//						numPENM, desiredPESizeNM, desiredTileSizeCM, desiredPESizeCM, CMTileheight, CMTilewidth, NMTileheight, NMTilewidth,
//						&layerReadLatency, &layerReadDynamicEnergy, &tileLeakage, &layerbufferLatency, &layerbufferDynamicEnergy, &layericLatency, &layericDynamicEnergy,
//						&coreLatencyADC, &coreLatencyAccum, &coreLatencyOther, &coreEnergyADC, &coreEnergyAccum, &coreEnergyOther, false, &layerclkPeriod);
//			if (param->synchronous) {
//				layerReadLatency *= clkPeriod;
//				layerbufferLatency *= clkPeriod;
//				layericLatency *= clkPeriod;
//				coreLatencyADC *= clkPeriod;
//				coreLatencyAccum *= clkPeriod;
//				coreLatencyOther *= clkPeriod;
//			}
//
//			double numTileOtherLayer = 0;
//			double layerLeakageEnergy = 0;
//			for (int j=0; j<netStructure.size(); j++) {
//				if (j != i) {
//					numTileOtherLayer += numTileEachLayer[0][j] * numTileEachLayer[1][j];
//				}
//			}
//			layerLeakageEnergy = numTileOtherLayer*layerReadLatency*tileLeakage;
//
//			cout << "layer" << i+1 << "'s readLatency is: " << layerReadLatency*1e9 << "ns" << endl;
//			myfile << "layer" << i+1 << "'s readLatency is: " << layerReadLatency*1e9 << "ns" << endl;
//			cout << "layer" << i+1 << "'s readLatency is: " << layerReadLatency*1e9 << "ns" << endl;
//			myfile << "layer" << i+1 << "'s readLatency is: " << layerReadLatency*1e9 << "ns" << endl;
//			cout << "layer" << i+1 << "'s readDynamicEnergy is: " << layerReadDynamicEnergy*1e12 << "pJ" << endl;
//			myfile << "layer" << i+1 << "'s readDynamicEnergy is: " << layerReadDynamicEnergy*1e12 << "pJ" << endl;
//			cout << "layer" << i+1 << "'s leakagePower is: " << numTileEachLayer[0][i] * numTileEachLayer[1][i] * tileLeakage*1e6 << "uW" << endl;
//			myfile << "layer" << i+1 << "'s leakagePower is: " << numTileEachLayer[0][i] * numTileEachLayer[1][i] * tileLeakage*1e6 << "uW" << endl;
//			cout << "layer" << i+1 << "'s leakageEnergy is: " << layerLeakageEnergy*1e12 << "pJ" << endl;
//			myfile << "layer" << i+1 << "'s leakageEnergy is: " << layerLeakageEnergy*1e12 << "pJ" << endl;
//			cout << "layer" << i+1 << "'s buffer latency is: " << layerbufferLatency*1e9 << "ns" << endl;
//			myfile << "layer" << i+1 << "'s buffer latency is: " << layerbufferLatency*1e9 << "ns" << endl;
//			cout << "layer" << i+1 << "'s buffer readDynamicEnergy is: " << layerbufferDynamicEnergy*1e12 << "pJ" << endl;
//			myfile << "layer" << i+1 << "'s buffer readDynamicEnergy is: " << layerbufferDynamicEnergy*1e12 << "pJ" << endl;
//			cout << "layer" << i+1 << "'s ic latency is: " << layericLatency*1e9 << "ns" << endl;
//			myfile << "layer" << i+1 << "'s ic latency is: " << layericLatency*1e9 << "ns" << endl;
//			cout << "layer" << i+1 << "'s ic readDynamicEnergy is: " << layericDynamicEnergy*1e12 << "pJ" << endl;
//			myfile << "layer" << i+1 << "'s ic readDynamicEnergy is: " << layericDynamicEnergy*1e12 << "pJ" << endl;
//
//
//			cout << endl;
//			myfile << endl;
//			cout << "************************ Breakdown of Latency and Dynamic Energy *************************" << endl;
//			myfile << "************************ Breakdown of Latency and Dynamic Energy *************************" << endl;
//			cout << endl;
//			myfile << endl;
//			cout << "----------- ADC (or S/As and precharger for SRAM) readLatency is : " << coreLatencyADC*1e9 << "ns" << endl;
//			myfile << "----------- ADC (or S/As and precharger for SRAM) readLatency is : " << coreLatencyADC*1e9 << "ns" << endl;
//			cout << "----------- Accumulation Circuits (subarray level: adders, shiftAdds; PE/Tile/Global level: accumulation units) readLatency is : " << coreLatencyAccum*1e9 << "ns" << endl;
//			myfile << "----------- Accumulation Circuits (subarray level: adders, shiftAdds; PE/Tile/Global level: accumulation units) readLatency is : " << coreLatencyAccum*1e9 << "ns" << endl;
//			cout << "----------- Other Peripheries (e.g. decoders, mux, switchmatrix, buffers, IC, pooling and activation units) readLatency is : " << coreLatencyOther*1e9 << "ns" << endl;
//			myfile << "----------- Other Peripheries (e.g. decoders, mux, switchmatrix, buffers, IC, pooling and activation units) readLatency is : " << coreLatencyOther*1e9 << "ns" << endl;
//			cout << "----------- ADC (or S/As and precharger for SRAM) readDynamicEnergy is : " << coreEnergyADC*1e12 << "pJ" << endl;
//			myfile << "----------- ADC (or S/As and precharger for SRAM) readDynamicEnergy is : " << coreEnergyADC*1e12 << "pJ" << endl;
//			cout << "----------- Accumulation Circuits (subarray level: adders, shiftAdds; PE/Tile/Global level: accumulation units) readDynamicEnergy is : " << coreEnergyAccum*1e12 << "pJ" << endl;
//			myfile << "----------- Accumulation Circuits (subarray level: adders, shiftAdds; PE/Tile/Global level: accumulation units) readDynamicEnergy is : " << coreEnergyAccum*1e12 << "pJ" << endl;
//			cout << "----------- Other Peripheries (e.g. decoders, mux, switchmatrix, buffers, IC, pooling and activation units) readDynamicEnergy is : " << coreEnergyOther*1e12 << "pJ" << endl;
//			myfile << "----------- Other Peripheries (e.g. decoders, mux, switchmatrix, buffers, IC, pooling and activation units) readDynamicEnergy is : " << coreEnergyOther*1e12 << "pJ" << endl;
//			cout << endl;
//			myfile << endl;
//			cout << "************************ Breakdown of Latency and Dynamic Energy *************************" << endl;
//			myfile << "************************ Breakdown of Latency and Dynamic Energy *************************" << endl;
//			cout << endl;
//			myfile << endl;
//
//			chipReadLatency += layerReadLatency;
//			chipReadDynamicEnergy += layerReadDynamicEnergy;
//			chipLeakageEnergy += layerLeakageEnergy;
//			chipLeakage += tileLeakage*numTileEachLayer[0][i] * numTileEachLayer[1][i];
//			chipbufferLatency += layerbufferLatency;
//			chipbufferReadDynamicEnergy += layerbufferDynamicEnergy;
//			chipicLatency += layericLatency;
//			chipicReadDynamicEnergy += layericDynamicEnergy;
//
//			chipLatencyADC += coreLatencyADC;
//			chipLatencyAccum += coreLatencyAccum;
//			chipLatencyOther += coreLatencyOther;
//			chipEnergyADC += coreEnergyADC;
//			chipEnergyAccum += coreEnergyAccum;
//			chipEnergyOther += coreEnergyOther;
//		}
//	} else {
//		// pipeline system
//		// firstly define system clock
//		double systemClock = 0;
//
//		vector<double> readLatencyPerLayer;
//		vector<double> readDynamicEnergyPerLayer;
//		vector<double> leakagePowerPerLayer;
//		vector<double> bufferLatencyPerLayer;
//		vector<double> bufferEnergyPerLayer;
//		vector<double> icLatencyPerLayer;
//		vector<double> icEnergyPerLayer;
//
//		vector<double> coreLatencyADCPerLayer;
//		vector<double> coreEnergyADCPerLayer;
//		vector<double> coreLatencyAccumPerLayer;
//		vector<double> coreEnergyAccumPerLayer;
//		vector<double> coreLatencyOtherPerLayer;
//		vector<double> coreEnergyOtherPerLayer;
//
//		for (int i=0; i<netStructure.size(); i++) {
//			ChipCalculatePerformance(inputParameter, tech, cell, i, argv[2*i+6], argv[2*i+6], argv[2*i+7], netStructure[i][6],
//						netStructure, markNM, numTileEachLayer, utilizationEachLayer, speedUpEachLayer, tileLocaEachLayer,
//						numPENM, desiredPESizeNM, desiredTileSizeCM, desiredPESizeCM, CMTileheight, CMTilewidth, NMTileheight, NMTilewidth,
//						&layerReadLatency, &layerReadDynamicEnergy, &tileLeakage, &layerbufferLatency, &layerbufferDynamicEnergy, &layericLatency, &layericDynamicEnergy,
//						&coreLatencyADC, &coreLatencyAccum, &coreLatencyOther, &coreEnergyADC, &coreEnergyAccum, &coreEnergyOther, false, &layerclkPeriod);
//			if (param->synchronous) {
//				layerReadLatency *= clkPeriod;
//				layerbufferLatency *= clkPeriod;
//				layericLatency *= clkPeriod;
//				coreLatencyADC *= clkPeriod;
//				coreLatencyAccum *= clkPeriod;
//				coreLatencyOther *= clkPeriod;
//			}
//
//			systemClock = MAX(systemClock, layerReadLatency);
//
//			readLatencyPerLayer.push_back(layerReadLatency);
//			readDynamicEnergyPerLayer.push_back(layerReadDynamicEnergy);
//			leakagePowerPerLayer.push_back(numTileEachLayer[0][i] * numTileEachLayer[1][i] * tileLeakage);
//			bufferLatencyPerLayer.push_back(layerbufferLatency);
//			bufferEnergyPerLayer.push_back(layerbufferDynamicEnergy);
//			icLatencyPerLayer.push_back(layericLatency);
//			icEnergyPerLayer.push_back(layericDynamicEnergy);
//
//			coreLatencyADCPerLayer.push_back(coreLatencyADC);
//			coreEnergyADCPerLayer.push_back(coreEnergyADC);
//			coreLatencyAccumPerLayer.push_back(coreLatencyAccum);
//			coreEnergyAccumPerLayer.push_back(coreEnergyAccum);
//			coreLatencyOtherPerLayer.push_back(coreLatencyOther);
//			coreEnergyOtherPerLayer.push_back(coreEnergyOther);
//		}
//
//		for (int i=0; i<netStructure.size(); i++) {
//
//			cout << "-------------------- Estimation of Layer " << i+1 << " ----------------------" << endl;
//			myfile << "-------------------- Estimation of Layer " << i+1 << " ----------------------" << endl;
//
//			cout << "layer" << i+1 << "'s readLatency is: " << readLatencyPerLayer[i]*1e9 << "ns" << endl;
//			myfile << "layer" << i+1 << "'s readLatency is: " << readLatencyPerLayer[i]*1e9 << "ns" << endl;
//			cout << "layer" << i+1 << "'s readDynamicEnergy is: " << readDynamicEnergyPerLayer[i]*1e12 << "pJ" << endl;
//			myfile << "layer" << i+1 << "'s readDynamicEnergy is: " << readDynamicEnergyPerLayer[i]*1e12 << "pJ" << endl;
//			cout << "layer" << i+1 << "'s leakagePower is: " << leakagePowerPerLayer[i]*1e6 << "uW" << endl;
//			myfile << "layer" << i+1 << "'s leakagePower is: " << leakagePowerPerLayer[i]*1e6 << "uW" << endl;
//			cout << "layer" << i+1 << "'s leakageEnergy is: " << leakagePowerPerLayer[i] * (systemClock-readLatencyPerLayer[i]) *1e12 << "pJ" << endl;
//			myfile << "layer" << i+1 << "'s leakageEnergy is: " << leakagePowerPerLayer[i] * (systemClock-readLatencyPerLayer[i]) *1e12 << "pJ" << endl;
//			cout << "layer" << i+1 << "'s buffer latency is: " << bufferLatencyPerLayer[i]*1e9 << "ns" << endl;
//			myfile << "layer" << i+1 << "'s buffer latency is: " << bufferLatencyPerLayer[i]*1e9 << "ns" << endl;
//			cout << "layer" << i+1 << "'s buffer readDynamicEnergy is: " << bufferEnergyPerLayer[i]*1e12 << "pJ" << endl;
//			myfile << "layer" << i+1 << "'s buffer readDynamicEnergy is: " << bufferEnergyPerLayer[i]*1e12 << "pJ" << endl;
//			cout << "layer" << i+1 << "'s ic latency is: " << icLatencyPerLayer[i]*1e9 << "ns" << endl;
//			myfile << "layer" << i+1 << "'s ic latency is: " << icLatencyPerLayer[i]*1e9 << "ns" << endl;
//			cout << "layer" << i+1 << "'s ic readDynamicEnergy is: " << icEnergyPerLayer[i]*1e12 << "pJ" << endl;
//			myfile << "layer" << i+1 << "'s ic readDynamicEnergy is: " << icEnergyPerLayer[i]*1e12 << "pJ" << endl;
//
//			cout << endl;
//			myfile << endl;
//			cout << "************************ Breakdown of Latency and Dynamic Energy *************************" << endl;
//			myfile << "************************ Breakdown of Latency and Dynamic Energy *************************" << endl;
//			cout << endl;
//			myfile << endl;
//			cout << "----------- ADC (or S/As and precharger for SRAM) readLatency is : " << coreLatencyADCPerLayer[i]*1e9 << "ns" << endl;
//			myfile << "----------- ADC (or S/As and precharger for SRAM) readLatency is : " << coreLatencyADCPerLayer[i]*1e9 << "ns" << endl;
//			cout << "----------- Accumulation Circuits (subarray level: adders, shiftAdds; PE/Tile/Global level: accumulation units) readLatency is : " << coreLatencyAccumPerLayer[i]*1e9 << "ns" << endl;
//			myfile << "----------- Accumulation Circuits (subarray level: adders, shiftAdds; PE/Tile/Global level: accumulation units) readLatency is : " << coreLatencyAccumPerLayer[i]*1e9 << "ns" << endl;
//			cout << "----------- Other Peripheries (e.g. decoders, mux, switchmatrix, buffers, IC, pooling and activation units) readLatency is : " << coreLatencyOtherPerLayer[i]*1e9 << "ns" << endl;
//			myfile << "----------- Other Peripheries (e.g. decoders, mux, switchmatrix, buffers, IC, pooling and activation units) readLatency is : " << coreLatencyOtherPerLayer[i]*1e9 << "ns" << endl;
//			cout << "----------- ADC (or S/As and precharger for SRAM) readDynamicEnergy is : " << coreEnergyADCPerLayer[i]*1e12 << "pJ" << endl;
//			myfile << "----------- ADC (or S/As and precharger for SRAM) readDynamicEnergy is : " << coreEnergyADCPerLayer[i]*1e12 << "pJ" << endl;
//			cout << "----------- Accumulation Circuits (subarray level: adders, shiftAdds; PE/Tile/Global level: accumulation units) readDynamicEnergy is : " << coreEnergyAccumPerLayer[i]*1e12 << "pJ" << endl;
//			myfile << "----------- Accumulation Circuits (subarray level: adders, shiftAdds; PE/Tile/Global level: accumulation units) readDynamicEnergy is : " << coreEnergyAccumPerLayer[i]*1e12 << "pJ" << endl;
//			cout << "----------- Other Peripheries (e.g. decoders, mux, switchmatrix, buffers, IC, pooling and activation units) readDynamicEnergy is : " << coreEnergyOtherPerLayer[i]*1e12 << "pJ" << endl;
//			myfile << "----------- Other Peripheries (e.g. decoders, mux, switchmatrix, buffers, IC, pooling and activation units) readDynamicEnergy is : " << coreEnergyOtherPerLayer[i]*1e12 << "pJ" << endl;
//			cout << endl;
//			myfile << endl;
//			cout << "************************ Breakdown of Latency and Dynamic Energy *************************" << endl;
//			myfile << "************************ Breakdown of Latency and Dynamic Energy *************************" << endl;
//			cout << endl;
//			myfile << endl;
//
//			chipReadLatency = systemClock;
//			chipReadDynamicEnergy += readDynamicEnergyPerLayer[i];
//			chipLeakageEnergy += leakagePowerPerLayer[i] * (systemClock-readLatencyPerLayer[i]);
//			chipLeakage += leakagePowerPerLayer[i];
//			chipbufferLatency = MAX(chipbufferLatency, bufferLatencyPerLayer[i]);
//			chipbufferReadDynamicEnergy += bufferEnergyPerLayer[i];
//			chipicLatency = MAX(chipicLatency, icLatencyPerLayer[i]);
//			chipicReadDynamicEnergy += icEnergyPerLayer[i];
//
//			chipLatencyADC = MAX(chipLatencyADC, coreLatencyADCPerLayer[i]);
//			chipLatencyAccum = MAX(chipLatencyAccum, coreLatencyAccumPerLayer[i]);
//			chipLatencyOther = MAX(chipLatencyOther, coreLatencyOtherPerLayer[i]);
//			chipEnergyADC += coreEnergyADCPerLayer[i];
//			chipEnergyAccum += coreEnergyAccumPerLayer[i];
//			chipEnergyOther += coreEnergyOtherPerLayer[i];
//		}
//
//	}
//
//	cout << "------------------------------ Summary --------------------------------" <<  endl;
//	myfile << "------------------------------ Summary --------------------------------" <<  endl;
//	cout << endl;
//	myfile << endl;
//	cout << "ChipArea : " << chipArea*1e12 << "um^2" << endl;
//	myfile << "ChipArea : " << chipArea*1e12 << "um^2" << endl;
//	cout << "Chip total CIM array : " << chipAreaArray*1e12 << "um^2" << endl;
//	myfile << "Chip total CIM array : " << chipAreaArray*1e12 << "um^2" << endl;
//	cout << "Total IC Area on chip (Global and Tile/PE local): " << chipAreaIC*1e12 << "um^2" << endl;
//	myfile << "Total IC Area on chip (Global and Tile/PE local): " << chipAreaIC*1e12 << "um^2" << endl;
//	cout << "Total ADC (or S/As and precharger for SRAM) Area on chip : " << chipAreaADC*1e12 << "um^2" << endl;
//	myfile << "Total ADC (or S/As and precharger for SRAM) Area on chip : " << chipAreaADC*1e12 << "um^2" << endl;
//	cout << "Total Accumulation Circuits (subarray level: adders, shiftAdds; PE/Tile/Global level: accumulation units) on chip : " << chipAreaAccum*1e12 << "um^2" << endl;
//	myfile << "Total Accumulation Circuits (subarray level: adders, shiftAdds; PE/Tile/Global level: accumulation units) on chip : " << chipAreaAccum*1e12 << "um^2" << endl;
//	cout << "Other Peripheries (e.g. decoders, mux, switchmatrix, buffers, pooling and activation units) : " << chipAreaOther*1e12 << "um^2" << endl;
//	myfile << "Other Peripheries (e.g. decoders, mux, switchmatrix, buffers, pooling and activation units) : " << chipAreaOther*1e12 << "um^2" << endl;
//	cout << endl;
//	myfile << endl;
//	if (! param->pipeline) {  //This is used
//		if (param->synchronous) {
//		cout << "Chip clock period is: " << clkPeriod*1e9 << "ns" <<endl;
//		myfile << "Chip clock period is: " << clkPeriod*1e9 << "ns" <<endl;
//		}
//		cout << "Chip layer-by-layer readLatency (per image) is: " << chipReadLatency*1e9 << "ns" << endl;
//		myfile << "Chip layer-by-layer readLatency (per image) is: " << chipReadLatency*1e9 << "ns" << endl;
//		cout << "Chip total readDynamicEnergy is: " << chipReadDynamicEnergy*1e12 << "pJ" << endl;
//		myfile << "Chip total readDynamicEnergy is: " << chipReadDynamicEnergy*1e12 << "pJ" << endl;
//		cout << "Chip total leakage Energy is: " << chipLeakageEnergy*1e12 << "pJ" << endl;
//		myfile << "Chip total leakage Energy is: " << chipLeakageEnergy*1e12 << "pJ" << endl;
//		cout << "Chip total leakage Power is: " << chipLeakage*1e6 << "uW" << endl;
//		myfile << "Chip total leakage Power is: " << chipLeakage*1e6 << "uW" << endl;
//		cout << "Chip buffer readLatency is: " << chipbufferLatency*1e9 << "ns" << endl;
//		myfile << "Chip buffer readLatency is: " << chipbufferLatency*1e9 << "ns" << endl;
//		cout << "Chip buffer readDynamicEnergy is: " << chipbufferReadDynamicEnergy*1e12 << "pJ" << endl;
//		myfile << "Chip buffer readDynamicEnergy is: " << chipbufferReadDynamicEnergy*1e12 << "pJ" << endl;
//		cout << "Chip ic readLatency is: " << chipicLatency*1e9 << "ns" << endl;
//		myfile << "Chip ic readLatency is: " << chipicLatency*1e9 << "ns" << endl;
//		cout << "Chip ic readDynamicEnergy is: " << chipicReadDynamicEnergy*1e12 << "pJ" << endl;
//		myfile << "Chip ic readDynamicEnergy is: " << chipicReadDynamicEnergy*1e12 << "pJ" << endl;
//	} else {
//		if (param->synchronous) {
//		cout << "Chip clock period is: " << clkPeriod*1e9 << "ns" <<endl;
//		myfile << "Chip clock period is: " << clkPeriod*1e9 << "ns" <<endl;
//		}
//		cout << "Chip pipeline-system-clock-cycle (per image) is: " << chipReadLatency*1e9 << "ns" << endl;
//		myfile << "Chip pipeline-system-clock-cycle (per image) is: " << chipReadLatency*1e9 << "ns" << endl;
//		cout << "Chip pipeline-system readDynamicEnergy (per image) is: " << chipReadDynamicEnergy*1e12 << "pJ" << endl;
//		myfile << "Chip pipeline-system readDynamicEnergy (per image) is: " << chipReadDynamicEnergy*1e12 << "pJ" << endl;
//		cout << "Chip pipeline-system leakage Energy (per image) is: " << chipLeakageEnergy*1e12 << "pJ" << endl;
//		myfile << "Chip pipeline-system leakage Energy (per image) is: " << chipLeakageEnergy*1e12 << "pJ" << endl;
//		cout << "Chip pipeline-system leakage Power (per image) is: " << chipLeakage*1e6 << "uW" << endl;
//		myfile << "Chip pipeline-system leakage Power (per image) is: " << chipLeakage*1e6 << "uW" << endl;
//		cout << "Chip pipeline-system buffer readLatency (per image) is: " << chipbufferLatency*1e9 << "ns" << endl;
//		myfile << "Chip pipeline-system buffer readLatency (per image) is: " << chipbufferLatency*1e9 << "ns" << endl;
//		cout << "Chip pipeline-system buffer readDynamicEnergy (per image) is: " << chipbufferReadDynamicEnergy*1e12 << "pJ" << endl;
//		myfile << "Chip pipeline-system buffer readDynamicEnergy (per image) is: " << chipbufferReadDynamicEnergy*1e12 << "pJ" << endl;
//		cout << "Chip pipeline-system ic readLatency (per image) is: " << chipicLatency*1e9 << "ns" << endl;
//		myfile << "Chip pipeline-system ic readLatency (per image) is: " << chipicLatency*1e9 << "ns" << endl;
//		cout << "Chip pipeline-system ic readDynamicEnergy (per image) is: " << chipicReadDynamicEnergy*1e12 << "pJ" << endl;
//		myfile << "Chip pipeline-system ic readDynamicEnergy (per image) is: " << chipicReadDynamicEnergy*1e12 << "pJ" << endl;
//	}
//
//	cout << endl;
//	myfile << endl;
//	cout << "************************ Breakdown of Latency and Dynamic Energy *************************" << endl;
//	myfile << "************************ Breakdown of Latency and Dynamic Energy *************************" << endl;
//	cout << endl;
//	myfile << endl;
//	cout << "----------- ADC (or S/As and precharger for SRAM) readLatency is : " << chipLatencyADC*1e9 << "ns" << endl;
//	myfile << "----------- ADC (or S/As and precharger for SRAM) readLatency is : " << chipLatencyADC*1e9 << "ns" << endl;
//	cout << "----------- Accumulation Circuits (subarray level: adders, shiftAdds; PE/Tile/Global level: accumulation units) readLatency is : " << chipLatencyAccum*1e9 << "ns" << endl;
//	myfile << "----------- Accumulation Circuits (subarray level: adders, shiftAdds; PE/Tile/Global level: accumulation units) readLatency is : " << chipLatencyAccum*1e9 << "ns" << endl;
//	cout << "----------- Other Peripheries (e.g. decoders, mux, switchmatrix, buffers, IC, pooling and activation units) readLatency is : " << chipLatencyOther*1e9 << "ns" << endl;
//	myfile << "----------- Other Peripheries (e.g. decoders, mux, switchmatrix, buffers, IC, pooling and activation units) readLatency is : " << chipLatencyOther*1e9 << "ns" << endl;
//	cout << "----------- ADC (or S/As and precharger for SRAM) readDynamicEnergy is : " << chipEnergyADC*1e12 << "pJ" << endl;
//	myfile << "----------- ADC (or S/As and precharger for SRAM) readDynamicEnergy is : " << chipEnergyADC*1e12 << "pJ" << endl;
//	cout << "----------- Accumulation Circuits (subarray level: adders, shiftAdds; PE/Tile/Global level: accumulation units) readDynamicEnergy is : " << chipEnergyAccum*1e12 << "pJ" << endl;
//	myfile << "----------- Accumulation Circuits (subarray level: adders, shiftAdds; PE/Tile/Global level: accumulation units) readDynamicEnergy is : " << chipEnergyAccum*1e12 << "pJ" << endl;
//	cout << "----------- Other Peripheries (e.g. decoders, mux, switchmatrix, buffers, IC, pooling and activation units) readDynamicEnergy is : " << chipEnergyOther*1e12 << "pJ" << endl;
//	myfile << "----------- Other Peripheries (e.g. decoders, mux, switchmatrix, buffers, IC, pooling and activation units) readDynamicEnergy is : " << chipEnergyOther*1e12 << "pJ" << endl;
//	cout << endl;
//	myfile << endl;
//	cout << "************************ Breakdown of Latency and Dynamic Energy *************************" << endl;
//	myfile << "************************ Breakdown of Latency and Dynamic Energy *************************" << endl;
//	cout << endl;
//	myfile << endl;
//
//	cout << endl;
//	myfile << endl;
//	cout << "----------------------------- Performance -------------------------------" << endl;
//	myfile << "----------------------------- Performance -------------------------------" << endl;
//	if (! param->pipeline) {
//		if(param->validated){
//			cout << "Energy Efficiency TOPS/W (Layer-by-Layer Process): " << numComputation/(chipReadDynamicEnergy*1e12+chipLeakageEnergy*1e12)/param->zeta << endl;	// post-layout energy increase, zeta = 1.23 by default
//			myfile << "Energy Efficiency TOPS/W (Layer-by-Layer Process): " << numComputation/(chipReadDynamicEnergy*1e12+chipLeakageEnergy*1e12)/param->zeta << endl;
//			cout << "Num Computations: " <<  numComputation << endl;
//			myfile << "Num Computations: " <<  numComputation << endl; // post-layout energy increase, zeta = 1.23 by default
//		}else{
//			cout << "Energy Efficiency TOPS/W (Layer-by-Layer Process): " << numComputation/(chipReadDynamicEnergy*1e12+chipLeakageEnergy*1e12) << endl;
//			myfile << "Energy Efficiency TOPS/W (Layer-by-Layer Process): " << numComputation/(chipReadDynamicEnergy*1e12+chipLeakageEnergy*1e12) << endl;
//		}
//		cout << "Throughput TOPS (Layer-by-Layer Process): " << numComputation/(chipReadLatency*1e12) << endl;
//		myfile << "Throughput TOPS (Layer-by-Layer Process): " << numComputation/(chipReadLatency*1e12) << endl;
//		cout << "Throughput FPS (Layer-by-Layer Process): " << 1/(chipReadLatency) << endl;
//		myfile << "Throughput FPS (Layer-by-Layer Process): " << 1/(chipReadLatency) << endl;
//		cout << "Compute efficiency TOPS/mm^2 (Layer-by-Layer Process): " << numComputation/(chipReadLatency*1e12)/(chipArea*1e6) << endl;
//		myfile << "Compute efficiency TOPS/mm^2 (Layer-by-Layer Process): " << numComputation/(chipReadLatency*1e12)/(chipArea*1e6) << endl;
//	} else {
//		if(param->validated){
//			cout << "Energy Efficiency TOPS/W (Pipelined Process): " << numComputation/(chipReadDynamicEnergy*1e12+chipLeakageEnergy*1e12)/param->zeta << endl;	// post-layout energy increase, zeta = 1.23 by default
//			myfile << "Energy Efficiency TOPS/W (Pipelined Process): " << numComputation/(chipReadDynamicEnergy*1e12+chipLeakageEnergy*1e12)/param->zeta << endl;	// post-layout energy increase, zeta = 1.23 by default
//		}else{
//			cout << "Energy Efficiency TOPS/W (Pipelined Process): " << numComputation/(chipReadDynamicEnergy*1e12+chipLeakageEnergy*1e12) << endl;
//			myfile << "Energy Efficiency TOPS/W (Pipelined Process): " << numComputation/(chipReadDynamicEnergy*1e12+chipLeakageEnergy*1e12) << endl;
//		}
//		cout << "Throughput TOPS (Pipelined Process): " << numComputation/(chipReadLatency*1e12) << endl;
//		myfile << "Throughput TOPS (Pipelined Process): " << numComputation/(chipReadLatency*1e12) << endl;
//		cout << "Throughput FPS (Pipelined Process): " << 1/(chipReadLatency) << endl;
//		myfile << "Throughput FPS (Pipelined Process): " << 1/(chipReadLatency) << endl;
//		cout << "Compute efficiency TOPS/mm^2 (Pipelined Process): " << numComputation/(chipReadLatency*1e12)/(chipArea*1e6) << endl;
//		myfile << "Compute efficiency TOPS/mm^2 (Pipelined Process): " << numComputation/(chipReadLatency*1e12)/(chipArea*1e6) << endl;
//	}
//	cout << "-------------------------------------- Hardware Performance Done --------------------------------------" <<  endl;
//	myfile << "-------------------------------------- Hardware Performance Done --------------------------------------" <<  endl;
//	cout << endl;
//	myfile << endl;
//	auto stop = chrono::high_resolution_clock::now();
//	auto duration = chrono::duration_cast<chrono::seconds>(stop-start);
//    cout << "------------------------------ Simulation Performance --------------------------------" <<  endl;
//    myfile << "------------------------------ Simulation Performance --------------------------------" <<  endl;
//	cout << "Total Run-time of NeuroSim: " << duration.count() << " seconds" << endl;
//	myfile << "Total Run-time of NeuroSim: " << duration.count() << " seconds" << endl;
//	cout << "------------------------------ Simulation Performance --------------------------------" <<  endl;
//	myfile << "------------------------------ Simulation Performance --------------------------------" <<  endl;
//    myfile.close();
//
//	return 0;
//}

vector<vector<double> > getNetStructure(const string &inputfile) {
	ifstream infile(inputfile.c_str());      
	string inputline;
	string inputval;
	
	int ROWin=0, COLin=0;      
	if (!infile.good()) {        
		cerr << "Error: the input file cannot be opened!" << endl;
		exit(1);
	}else{
		while (getline(infile, inputline, '\n')) {       
			ROWin++;                                
		}
		infile.clear();
		infile.seekg(0, ios::beg);      
		if (getline(infile, inputline, '\n')) {        
			istringstream iss (inputline);      
			while (getline(iss, inputval, ',')) {       
				COLin++;
			}
		}	
	}
	infile.clear();
	infile.seekg(0, ios::beg);          

	vector<vector<double> > netStructure;               
	for (int row=0; row<ROWin; row++) {	
		vector<double> netStructurerow;
		getline(infile, inputline, '\n');             
		istringstream iss;
		iss.str(inputline);
		for (int col=0; col<COLin; col++) {       
			while(getline(iss, inputval, ',')){	
				istringstream fs;
				fs.str(inputval);
				double f=0;
				fs >> f;				
				netStructurerow.push_back(f);			
			}			
		}		
		netStructure.push_back(netStructurerow);
	}
	infile.close();
	
	return netStructure;
	netStructure.clear();
}	



