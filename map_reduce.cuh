#ifndef HEADER_FILE
#define HEADER_FILE

// Code is based on the explanation of: 
// https://whatsbigdata.be/author/admin/
// https://www.youtube.com/watch?v=UzR7XB74tuk&list=WL&index=8&t=0s
// https://stackoverflow.com/questions/7042014/how-do-you-make-a-pair-vector-out-of-two-arrays-and-then-sort-by-the-first-eleme
// https://leimao.github.io/blog/Pass-Function-Pointers-to-Kernels-CUDA/
// https://saurzcode.in/2018/05/how-to-use-multithreadedmapper-in-mapreduce/
// http://www.learncertification.com/study-material/void-data-type-in-c
// https://stackoverflow.com/questions/33425304/how-to-pass-a-multidimensional-array-to-a-function-without-inner-dimension-in-c


#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <thrust/sort.h>
#include <math.h>


template<typename Key>
struct BinaryCompare {
	__host__ __device__ bool operator() (Key& lhs, Key& rhs) {  // From https://devtalk.nvidia.com/default/topic/419198/thrust-sort-need-some-help/
		// Take the byte representation of the keys
		void* void_lhs = (unsigned char*)&(lhs);
		void* void_rhs = (unsigned char*)&(rhs);
		// Go over all bytes
		for (int i = 0; i < sizeof(Key); i++) {
			unsigned char* p1 = (unsigned char*)void_lhs + i;
			unsigned char* p2 = (unsigned char*)void_rhs + i;
			// If the byte of key 1 is smaller than the one from key 2
			if (*p1 < *p2) {
				return true;
			}
			else if (*p1 > * p2) {
				return false;  // This varianble is necessary when the empty values defined by the user are bigger than the last key value, else it would be skipped.
			}
		}
		return false;
	}
};

template<typename Input, typename Key, typename Value>
__global__ void map_( Input* input,  int input_size, Key* keys, Value* values,  int amount_pairs_per_map, void(*mapper) (Input*, Key*, Value*, int))
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	while (i < input_size) {  // Every input needs to be mapped
		(*mapper)(input + i, keys + i * amount_pairs_per_map, values + i * amount_pairs_per_map, amount_pairs_per_map);  // Give the mapper the right place in the memory

		i += blockDim.x * gridDim.x;  // Go the total amount of threads further, important when the input is bigger than the amount of threads
	}

}

template<typename Value>
__global__ void reduce_(Value* values, int amount_same_key, void(*reducer) (Value*, Value*, Value*))
{
	int g_index = (threadIdx.x + (blockIdx.x * blockDim.x)) * 2;
	extern __shared__ Value s_data[];

	// Add the right part of the array to the shared memory
	if (g_index < amount_same_key) {
		if (g_index + 1 >= amount_same_key) {
			s_data[threadIdx.x] = values[g_index];
		}
		else {
			(*reducer)(values + g_index, values + g_index + 1, s_data + threadIdx.x);
		}
	}

	__syncthreads();

	// Check the size of the array that has been given to this block
	int size_array_in_this_block = blockDim.x;
	int rest = 0; // For odd sizes
	if (blockIdx.x == gridDim.x - 1) {
		size_array_in_this_block = ((amount_same_key - (gridDim.x - 1) * blockDim.x * 2) + 1) / 2; // Last array can have a shorter array than the blockSize
		rest = size_array_in_this_block % 2;  // Check if that size is odd or not
	}

	for (unsigned int i = size_array_in_this_block / 2; i > 0; i /= 2) {
		if (threadIdx.x < i) {
			// Add the elements in such an order that we don't have bank conflicts
			(*reducer)(s_data + threadIdx.x, s_data + threadIdx.x + i, s_data + threadIdx.x);
		}
		__syncthreads();
		if (blockIdx.x == gridDim.x - 1 && threadIdx.x == 0) {  // Only possible in last block (all the rest will have full even blockSize arrays)
			if (rest == 1) {  // If the size of the array is odd, do one extra reduce of the first and last element
				(*reducer)(s_data, s_data + i * 2, s_data);
			}
			rest = i % 2;
		}
		__syncthreads();
	}

	// Save the data at the right location
	if (threadIdx.x == 0) {
		values[blockIdx.x] = s_data[0];
	}
}


// Returns the number of output pairs that are used. Keys will be compared with a bytewise <.
template<typename Input, typename Key, typename Value>
int MapReduce( Input* input,  int input_size, Key* output_keys, Value* output_values, int output_size,  int amount_pairs_per_map, void(*mapper) (Input*, Key*, Value*, int), void(*reducer) (Value*, Value*, Value*), bool print_warning = true)
{
	printf("Map reduce start...\n");
	// Copy input to cuda mem
	Input* input_mem;
	cudaMallocManaged(&input_mem, input_size * input_size);
	std::copy(input, input + input_size, input_mem);

	// Allocate mem for keys and values, aka the pairs
	 int amount_of_pairs = input_size * amount_pairs_per_map;
	Key* keys;
	cudaMallocManaged(&keys, amount_of_pairs * sizeof(Key));  // You don't know how much keys, the amount can be different for every piece of the input
	Value* values;
	cudaMallocManaged(&values, amount_of_pairs * sizeof(Value));

	// Map with mapper function
	int num_blocks = (input_size / 1024) + 1;  // +1 to make sure it isn't 0
	if (num_blocks > 65536) {
		num_blocks = 65536;
	}
	printf("Map...\n");
	map_<Input, Key, Value> << <num_blocks, 1024 >>>(input_mem, input_size, keys, values, amount_pairs_per_map, mapper);
	cudaDeviceSynchronize();

	// Release input mem
	cudaFree(input_mem);

	// Shuffle/sort operation with respect to the key in the pairs
	printf("Sort...\n");
	thrust::sort_by_key(keys, keys + amount_of_pairs, values, BinaryCompare<Key>());
	cudaDeviceSynchronize();

	// Reduce with reducer function
	printf("Reduce...\n");
	// Initalize indexers and kernel numbers
	int nr_key = 0;
	int old_end = 0;
	// Go over all the pairs and execute the reduction kernel per key group (= subarray with all the pairs with the same key)
	for (int j = 0; j < amount_of_pairs; j++) {
		// If the next key is different or it's the last pair, the key group ends
		if (j == amount_of_pairs - 1 || BinaryCompare<Key>()(keys[j], keys[j + 1])) {
			// The reduction algorithm on the key group
			int numBlocks = 0;
			int blockSize = 1024;
			for (int i = j - old_end + 1; i > 1; i = numBlocks) {
				if ((i + 1) / 2 > blockSize) {  // When multiple blocks are used
					numBlocks = ((i + blockSize * 2) / (blockSize * 2)); // # threads = # elements / 2 (a reduce is performed during the loading of the elements in the shared mem)
				}
				else {  // When only one block is used
					blockSize = (i + 1) / 2; // (int)ceil((double)i / (double)2.0);
					numBlocks = 1;
				}
				// Execute the reducer on the key group
				reduce_ <Value> << <numBlocks, blockSize, blockSize * sizeof(Value) >> > (values + old_end, i, reducer);
				cudaDeviceSynchronize();
			}
			// Save the result of the key group as one output
			output_keys[nr_key] = keys[j];
			output_values[nr_key] = values[old_end];
			// Raise indexers
			old_end = j + 1;
			nr_key++;
			if (nr_key == output_size) {  // When the ouptut size is reached, stop
				printf("!!!The output size has been reached!!!  (Turn this warning off by adding false as an argument to the mapreduce function)\n-> %i of the %i pairs are used.\n", j, amount_of_pairs);
				break;
			}
		}
	}
	if (nr_key < output_size) {  // When the ouptut size is reached, stop
		printf("!!!All pairs have been used!!!  (Turn this warning off by adding false as an argument to the mapreduce function)\n-> %i of the %i outputs are filled.\n", nr_key, output_size);
	}

	// Release cuda mem
	cudaFree(keys);
	cudaFree(values);

	printf("Map reduce end.\n");

	return nr_key;  // Return how many outputs are filled
}

#endif
