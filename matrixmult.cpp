#include <amp.h>
#include <iostream>
#include <assert.h>

using namespace concurrency;
#define USE_GPU 1
#include "matcalc.hpp"

class measurement_time
{
	std::chrono::system_clock::time_point start_;
	std::chrono::system_clock::time_point end;
public:
	measurement_time()
	{
		start_ = std::chrono::system_clock::now();
	}

	inline void start()
	{
		start_ = std::chrono::system_clock::now();
	}
	inline void stop()
	{
		end = std::chrono::system_clock::now();  // åvë™èIóπéûä‘

		double elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start_).count();
		printf("%f[milliseconds]\n", elapsed);
	}

};

#define A	0
#if A
#define DATA_TYPE float
#else
#define DATA_TYPE double
#endif

//----------------------------------------------------------------------------
// Generate random data
//----------------------------------------------------------------------------
template<typename _type>
void initialize_array(std::vector<_type> &v_data, unsigned size)
{
    for(unsigned i=0; i<size; ++i)
    {
        v_data[i] = (_type)((_type)rand() * 1 / (_type)(RAND_MAX + 1));
    }
}


template<typename _type>
bool verify(std::vector<_type>& v_res, std::vector<_type>& v_ref, int len)
{
    bool passed = true;

    for (int i = 0; i < len; ++i)
    {
        if (v_res[i] != v_ref[i])
        {
             printf("v_res[%d] = %f, v_ref[%d] = %f\n", i, v_res[i], i, v_ref[i]);
             passed = false;
             break;
        }
    }

    return passed;
}

template<>
bool verify(std::vector<float>& v_res, std::vector<float>& v_ref, int len)
{
    bool passed = true;

    for (int i = 0; i < len; ++i)
    {
        if (fabs(v_res[i] - v_ref[i]) > 0.01)
        {
             printf("v_res[%d] = %f, v_ref[%d] = %f\n", i, v_res[i], i, v_ref[i]);
             passed = false;
             break;
        }
    }

    return passed;
}

template<>
bool verify(std::vector<double>& v_res, std::vector<double>& v_ref, int len)
{
    bool passed = true;

    for (int i = 0; i < len; ++i)
    {
        if (fabs(v_res[i] - v_ref[i]) > 0.01)
        {
             printf("v_res[%d] = %f, v_ref[%d] = %f\n", i, v_res[i], i, v_ref[i]);
             passed = false;
             break;
        }
    }

    return passed;
}


int main()
{
    accelerator default_device;
    std::wcout << L"Using device : " << default_device.get_description() << std::endl;
    if (default_device == accelerator(accelerator::direct3d_ref))
        std::cout << "WARNING!! Running on very slow emulator! Only use this accelerator for debugging." << std::endl;

    srand(2012);

    const int M = 256 * 2/2;
    const int N = 256 * 2/2;
    const int W = 256 * 2/2;
    
    std::vector<DATA_TYPE> v_a(M * N);
    std::vector<DATA_TYPE> v_b(N * W);
    std::vector<DATA_TYPE> v_c_simple(M * W);
    std::vector<DATA_TYPE> v_c_tiled(M * W);
    std::vector<DATA_TYPE> v_ref(M * W);

    initialize_array(v_a, M * N);
    initialize_array(v_b, N * W);

    assert((M!=0) && (W!=0) && (N!=0));

    printf("Matrix dimension C(%d x %d) = A(%d x %d) * B(%d x %d)\n", M, W, M, N, N, W);

	measurement_time me;

	printf("cpu Simple\n");
	me.start();
	mull_standerd0(&v_a[0], M, N, &v_b[0], N, W, &v_ref[0]);
	printf("completed.\n");
	me.stop();
	printf("------------------------------------------------------------\n\n");

	printf("multi core_cpu Simple\n");
	me.start();
	std::vector<DATA_TYPE> v_ref2(M * W);
	mull_standerd(&v_a[0], M, N, &v_b[0], N, W, &v_ref2[0]);
	printf("completed.\n");
	printf("\t%s\n\n", verify(v_ref2, v_ref, M * W) ? "Data matches" : "Data mismatch");
	me.stop();
	printf("------------------------------------------------------------\n\n");

	printf("multi core_cpu Unrolling Simple\n");
	me.start();
	std::vector<DATA_TYPE> v_ref4(M * W);
	mull_Unrolling(&v_a[0], M, N, &v_b[0], N, W, &v_ref4[0]);
	printf("completed.\n");
	printf("\t%s\n\n", verify(v_ref4, v_ref, M * W) ? "Data matches" : "Data mismatch");
	me.stop();
	printf("------------------------------------------------------------\n\n");

	printf("amp Simple\n");
	me.start();
	std::vector<DATA_TYPE> v_ref3(M * W);
	mull_gpu(&v_a[0], M, N, &v_b[0], N, W, &v_ref3[0]);
	printf(" completed.\n");
	printf("\t%s\n\n", verify(v_ref3, v_ref, M * W) ? "Data matches" : "Data mismatch");
	me.stop();
	printf("------------------------------------------------------------\n\n");

	printf("amp tiled\n");
	me.start();
	std::vector<DATA_TYPE> v_ref5(M * W);
	mull_gpu_tiled<DATA_TYPE,16>(&v_a[0], M, N, &v_b[0], N, W, &v_ref5[0]);
	printf(" completed.\n");
	printf("\t%s\n\n", verify(v_ref5, v_ref, M * W) ? "Data matches" : "Data mismatch");
	me.stop();
	printf("------------------------------------------------------------\n\n");

    return 0;
}
