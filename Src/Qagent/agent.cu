/*************************************************************************
/* ECE 277: GPU Programmming 2021 WINTER
/* Author and Instructer: Cheolhong An
/* Copyright 2019
/* University of California, San Diego
/*************************************************************************/
#include <cuda_fp16.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <helper_functions.h>
#include <curand.h>
#include <curand_kernel.h>
#include <cmath>

float epsilon;

float* d_epsilon;
short* d_action;
float* d_qtable; // 32 * 32 * 4; 2d grid, 1d block; shared d_qtable[idx] stores the corresponding q value
curandState* d_randstate;
short* d_active;

__global__ void Action_init(short* d_action) {
	unsigned int idx = (gridDim.y * blockIdx + blockIdx.x ) * blockDim.x + threadIdx.x;
	d_action[idx] = 0;
}

__global__ void Active_init(short* d_active) {
	unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
	d_active[idx] = 1;
}

__global__ void Qtable_init(float* d_qtable) {
	unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
	d_qtable[idx] = 0;
}

__global__ void Randstate_init(curandState* d_randstate) {
	unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
	curand_init(clock() + idx, idx, 0, &d_randstate[idx]);
}


void agent_init()
{
	// add your codes
	cudaMalloc((void**)&d_qtable, sizeof(float) * 32 * 32 * 4);
	cudaMalloc((void**)&d_epsilon, sizeof(float) * 1);
	cudaMalloc((void**)&d_action, sizeof(short) * 128);
	cudaMalloc((void**)&d_randstate, sizeof(curandState) * 128);
	cudaMalloc((void**)&d_active, sizeof(short) * 128);

	epsilon = 1.000f;
	cudaMemcpy(d_epsilon, &epsilon, sizeof(float), cudaMemcpyHostToDevice);

	dim3 grid(32, 32);
	dim3 block(4);

	Qtable_init << <grid, block >> > (d_qtable);
	Randstate_init << <1, 128 >> > (d_randstate);
	Action_init << <1, 128 >> > (d_action);
	Active_init << <1, 128 >> > (d_active);
}

void agent_init_episode() {
	Active_init << <1, 128 >> > (d_active);
}

__global__ void Agent_action(int2* cstate, curandState* d_randstate, float* d_epsilon, short* d_action, float* d_qtable, short* d_active) {
	unsigned int agent_id = blockIdx.x * blockDim.x + threadIdx.x;
	float uni_dist = curand_uniform(&d_randstate[agent_id]);

	if (d_active[agent_id] != 1) {
		return;
	}

	if (uni_dist < d_epsilon[0]) {
		// exploration: 0 ~ 0.25, 0.25 ~ 0.5, 0.5 ~ 0.75, 0.75 ~ 1
		// d_epsilon < 1, need to create curand_uniform
		// ceil, uni_dist may be 1
		if (d_epsilon[0] < 1) {
			uni_dist = curand_uniform(&d_randstate[agent_id]);
		}
		d_action[agent_id] = (short)ceil(uni_dist * 4) - 1;
	}
	else {
		// exploitation
		int q_id = cstate[agent_id].y * 32 * 4 + cstate[agent_id].x * 4;
		float q_max = d_qtable[q_id];
		d_action[agent_id] = 0;

		for (short i = 1; i <= 3; i++) {
			if (d_qtable[q_id + i] > q_max) {
				q_max = d_qtable[q_id + i];
				d_action[agent_id] = i;
			}
		}
	}
}

short* agent_action(int2* cstate)
{
	// add your codes
	Agent_action << <1, 128 >> > (cstate, d_randstate, d_epsilon, d_action, d_qtable, d_active);
	return d_action;
}

__global__ void Agent_update(int2* cstate, int2* nstate, float* rewards, float* d_qtable, short* d_active) {
	unsigned int agent_id = blockIdx.x * blockDim.x + threadIdx.x;
	int q_cur_id = cstate[agent_id].y * 32 * 4 + cstate[agent_id].x * 4;
	int q_next_id = nstate[agent_id].y * 32 * 4 + nstate[agent_id].x * 4;
	float q_next_max = d_qtable[q_next_id];

	if (rewards[agent_id] != 0) {
		d_active[agent_id] = 0;
	}

	if (d_active[agent_id] != 1) {
		return;
	}

	for (short i = 1; i <= 3; i++) {
		if (d_qtable[q_next_id + i] > q_next_max) {
			q_next_max = d_qtable[q_next_id + i];
		}
	}

	//step flag, no need to 0.9 * q_next_max
	d_qtable[q_cur_id] += 0.1 * (rewards[agent_id] + 0.9 * q_next_max - d_qtable[q_cur_id]);
	if (rewards[agent_id] != 0) {
		d_active[agent_id] = 0;
	}

	if (d_active[agent_id] != 1) {
		return;
	}
}

void agent_update(int2* cstate, int2* nstate, float *rewards)
{
	// add your codes
	Agent_update << <1, 128 >> > (cstate, nstate, rewards, d_qtable, d_active);
}



__global__ void Adjust_epsilon(float* d_epsilon) {
	d_epsilon[0] -= 0.001;
	if (d_epsilon[0] < 0.1) {
		d_epsilon[0] = 0.1;
	}
}

float agent_adjustepsilon()
{
	// add your codes
	Adjust_epsilon << <1, 1 >> > (d_epsilon);
	cudaMemcpy(&epsilon, d_epsilon, sizeof(float), cudaMemcpyDeviceToHost);

	return epsilon;
}


3
