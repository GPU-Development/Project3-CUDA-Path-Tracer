#include <cstdio>
#include <cuda.h>
#include <cmath>
#include <chrono>
#include <thrust/execution_policy.h>
#include <thrust/random.h>
#include <thrust/remove.h>
#include <thrust/copy.h>
#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>

#include "stream_compaction\efficient.h"
#include "sceneStructs.h"
#include "scene.h"
#include "glm/glm.hpp"
#include "glm/gtx/norm.hpp"
#include "utilities.h"
#include "pathtrace.h"
#include "intersections.h"
#include "interactions.h"


#define SORT_BY_MATERIAL 0
#define CACHE_FIRST_BOUNCE 1
#define ANTIALIASING 1
#define CUSTOM_COMPACT 0
#define DIRECT_LIGHTING 1

#define BLOCK_SIZE 128
#define ERRORCHECK 1

#define FILENAME (strrchr(__FILE__, '/') ? strrchr(__FILE__, '/') + 1 : __FILE__)
#define checkCUDAError(msg) checkCUDAErrorFn(msg, FILENAME, __LINE__)
void checkCUDAErrorFn(const char *msg, const char *file, int line) {
#if ERRORCHECK
    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (cudaSuccess == err) {
        return;
    }

    fprintf(stderr, "CUDA error");
    if (file) {
        fprintf(stderr, " (%s:%d)", file, line);
    }
    fprintf(stderr, ": %s: %s\n", msg, cudaGetErrorString(err));
#  ifdef _WIN32
    getchar();
#  endif
    exit(EXIT_FAILURE);
#endif
}

__host__ __device__
thrust::default_random_engine makeSeededRandomEngine(int iter, int index, int depth) {
    int h = utilhash((1 << 31) | (depth << 22) | iter) ^ utilhash(index);
    return thrust::default_random_engine(h);
}

//Kernel that writes the image to the OpenGL PBO directly.
__global__ void sendImageToPBO(uchar4* pbo, glm::ivec2 resolution,
        int iter, glm::vec3* image) {
    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;

    if (x < resolution.x && y < resolution.y) {
        int index = x + (y * resolution.x);
        glm::vec3 pix = image[index];

        glm::ivec3 color;
        color.x = glm::clamp((int) (pix.x / iter * 255.0), 0, 255);
        color.y = glm::clamp((int) (pix.y / iter * 255.0), 0, 255);
        color.z = glm::clamp((int) (pix.z / iter * 255.0), 0, 255);

        // Each thread writes one pixel location in the texture (textel)
        pbo[index].w = 0;
        pbo[index].x = color.x;
        pbo[index].y = color.y;
        pbo[index].z = color.z;
    }
}

static Scene * hst_scene = NULL;
static glm::vec3 * dev_image = NULL;
static Geom * dev_geoms = NULL;
static Material * dev_materials = NULL;
static PathSegment * dev_paths = NULL;
static ShadeableIntersection * dev_intersections = NULL;
// TODO: static variables for device memory, any extra info you need, etc
// ...
static PathSegment * dev_first_paths = NULL;
static ShadeableIntersection * dev_first_intersection = NULL;
static Geom * dev_lights = NULL;
static int * dev_active = NULL;
static int * dev_inactive = NULL;
static int * dev_compacted = NULL;

void pathtraceInit(Scene *scene) {
    hst_scene = scene;
    const Camera &cam = hst_scene->state.camera;
    const int pixelcount = cam.resolution.x * cam.resolution.y;

    cudaMalloc(&dev_image, pixelcount * sizeof(glm::vec3));
    cudaMemset(dev_image, 0, pixelcount * sizeof(glm::vec3));

  	cudaMalloc(&dev_paths, pixelcount * sizeof(PathSegment));

  	cudaMalloc(&dev_geoms, scene->geoms.size() * sizeof(Geom));
  	cudaMemcpy(dev_geoms, scene->geoms.data(), scene->geoms.size() * sizeof(Geom), cudaMemcpyHostToDevice);

  	cudaMalloc(&dev_materials, scene->materials.size() * sizeof(Material));
  	cudaMemcpy(dev_materials, scene->materials.data(), scene->materials.size() * sizeof(Material), cudaMemcpyHostToDevice);

  	cudaMalloc(&dev_intersections, pixelcount * sizeof(ShadeableIntersection));
  	cudaMemset(dev_intersections, 0, pixelcount * sizeof(ShadeableIntersection));

    // TODO: initialize any extra device memeory you need
	cudaMalloc(&dev_first_paths, pixelcount * sizeof(PathSegment));

	cudaMalloc(&dev_first_intersection, pixelcount * sizeof(ShadeableIntersection));
	cudaMemset(dev_first_intersection, 0, pixelcount * sizeof(ShadeableIntersection));

	cudaMalloc(&dev_lights, scene->geoms.size() * sizeof(Geom));

	cudaMalloc(&dev_active, pixelcount * sizeof(int));
	cudaMemset(dev_active, 0, pixelcount * sizeof(int));

	cudaMalloc(&dev_inactive, pixelcount * sizeof(int));
	cudaMemset(dev_inactive, 0, pixelcount * sizeof(int));

	cudaMalloc(&dev_compacted, pixelcount * sizeof(int));
	cudaMemset(dev_compacted, 0, pixelcount * sizeof(int));

    checkCUDAError("pathtraceInit");
}

void pathtraceFree() {
    cudaFree(dev_image);  // no-op if dev_image is null
  	cudaFree(dev_paths);
  	cudaFree(dev_geoms);
  	cudaFree(dev_materials);
  	cudaFree(dev_intersections);
    // TODO: clean up any extra device memory you created
	cudaFree(dev_first_paths);
	cudaFree(dev_first_intersection);
	cudaFree(dev_lights);
	cudaFree(dev_active);
	cudaFree(dev_inactive);
	cudaFree(dev_compacted);

    checkCUDAError("pathtraceFree");
}

/**
* Generate PathSegments with rays from the camera through the screen into the
* scene, which is the first bounce of rays.
*
* Antialiasing - add rays for sub-pixel sampling
* motion blur - jitter rays "in time"
* lens effect - jitter ray origin positions based on a lens
*/
__global__ void generateRayFromCamera(Camera cam, int iter, int traceDepth, PathSegment* pathSegments)
{
	int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	int y = (blockIdx.y * blockDim.y) + threadIdx.y;

	if (x < cam.resolution.x && y < cam.resolution.y) {
		int index = x + (y * cam.resolution.x);
		PathSegment & segment = pathSegments[index];

		segment.ray.origin = cam.position;
		segment.color = glm::vec3(1.0f, 1.0f, 1.0f);

#if ANTIALIASING
		thrust::default_random_engine rng = makeSeededRandomEngine(index, iter, 0);
		thrust::uniform_real_distribution<float> u01(0, 1);

		glm::vec3 jitter((u01(rng) - 0.5f) * cam.pixelLength.x, (u01(rng) - 0.5f) * cam.pixelLength.y, 0);

		segment.ray.direction = glm::normalize(cam.view
			- cam.right * cam.pixelLength.x * ((float)x - (float)cam.resolution.x * 0.5f)
			- cam.up * cam.pixelLength.y * ((float)y - (float)cam.resolution.y * 0.5f) 
			+ jitter
			);
#else
		segment.ray.direction = glm::normalize(cam.view
			- cam.right * cam.pixelLength.x * ((float)x - (float)cam.resolution.x * 0.5f)
			- cam.up * cam.pixelLength.y * ((float)y - (float)cam.resolution.y * 0.5f)
			);
#endif

		segment.pixelIndex = index;
		segment.remainingBounces = traceDepth;
		segment.ray.indexOfRefraction = 1.0f;
	}
}

__global__ void pathTraceOneBounce(
	int depth
	, int num_paths
	, PathSegment * pathSegments
	, Geom * geoms
	, int geoms_size
	, ShadeableIntersection * intersections
	)
{
	int path_index = blockIdx.x * blockDim.x + threadIdx.x;
	PathSegment pathSegment;

	if (path_index < num_paths)
	{
		pathSegment = pathSegments[path_index];

		float t;
		glm::vec3 intersect_point;
		glm::vec3 normal;
		float t_min = FLT_MAX;
		int hit_geom_index = -1;
		bool outside = true;
		
		bool tmp_outside = true;
		glm::vec3 tmp_intersect;
		glm::vec3 tmp_normal;

		// naive parse through global geoms
		for (int i = 0; i < geoms_size; i++)
		{
			Geom & geom = geoms[i];

			if (geom.type == CUBE)
			{
				t = boxIntersectionTest(geom, pathSegment.ray, tmp_intersect, tmp_normal, tmp_outside);
			}
			else if (geom.type == SPHERE)
			{
				t = sphereIntersectionTest(geom, pathSegment.ray, tmp_intersect, tmp_normal, tmp_outside);
			}
			// TODO: add more intersection tests here... triangle? metaball? CSG?

			// Compute the minimum t from the intersection tests to determine what
			// scene geometry object was hit first.
			if (t > 0.0f && t_min > t)
			{
				t_min = t;
				hit_geom_index = i;
				intersect_point = tmp_intersect;
				normal = tmp_normal;
				outside = tmp_outside;
			}
		}

		if (hit_geom_index == -1)
		{
			intersections[path_index].t = -1.0f;
		}
		else
		{
			//The ray hits something
			intersections[path_index].t = t_min;
			intersections[path_index].materialId = geoms[hit_geom_index].materialid;
			intersections[path_index].surfaceNormal = normal;
			intersections[path_index].outside = outside;
		}
	}
}

__device__ PathSegment computeNewRay(int i, Material &material, PathSegment &currentPath, ShadeableIntersection &intersection)
{
	// Set up the RNG
	thrust::default_random_engine rng = makeSeededRandomEngine(currentPath.remainingBounces + currentPath.pixelIndex, i, intersection.materialId);

	// Create new path
	PathSegment newPath = currentPath;
	scatterRay(newPath.ray, newPath.color, intersection, material, rng);
	newPath.remainingBounces = currentPath.remainingBounces - 1;

	return newPath;
}


__global__ void shadeMaterialSimple(
	int iter,
	int traceDepth,
	int num_paths,
	ShadeableIntersection * shadeableIntersections,
	PathSegment * pathSegments,
	Material * materials
	)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < num_paths)
	{
		PathSegment &currentRay = pathSegments[idx];
		ShadeableIntersection intersection = shadeableIntersections[idx];

		if (intersection.t > 0.0f) { // if the intersection exists...
			Material &material = materials[intersection.materialId];
			glm::vec3 materialColor = material.color;

			// If the material indicates that the object was a light, "light" the ray
			if (material.emittance > 0.0f) {
				if (currentRay.remainingBounces > 0) {
					currentRay.color *= (materialColor * material.emittance);
				}
				currentRay.remainingBounces = 0.0f;
			} else if (currentRay.remainingBounces > 0) {;
				currentRay = computeNewRay(iter, material, currentRay, intersection);
			} else {
				currentRay.color = glm::vec3(0.0f); // Bottomed out without hitting a light
			}


		}
		else {
			currentRay.color = glm::vec3(0.0f);
			currentRay.remainingBounces = -1.0f;
		}
	}
}

__global__ void directLighting(
	int iter,
	int num_paths,
	ShadeableIntersection * shadeableIntersections,
	PathSegment * pathSegments,
	Material * materials,
	Geom * geoms,
	int geoms_size,
	Geom * lights,
	int lights_size
	) 
{
	int path_index = blockIdx.x * blockDim.x + threadIdx.x;
	PathSegment pathSegment;
	

	if (path_index < num_paths)
	{
		PathSegment &pathSegment = pathSegments[path_index];

		// choose random light
		int l = -1;
		thrust::default_random_engine rng = makeSeededRandomEngine(path_index, iter, l);
		thrust::uniform_real_distribution<float> u01(0, 1);
		l = u01(rng)*lights_size;
			
		// generate a new ray to a random position on each light
		Geom & geom = lights[l];
		glm::vec3 lightSample(u01(rng)*2.0f - 1.0f, u01(rng)*2.0f - 1.0f, u01(rng)*2.0f - 1.0f);
		if (geom.type == CUBE)
		{
			lightSample = glm::normalize(lightSample) * geom.scale;
		}
		else if (geom.type == SPHERE)
		{
			lightSample *= geom.scale;
		}
		lightSample += geom.translation;
		pathSegment.ray.direction = glm::normalize(lightSample - pathSegment.ray.origin);
		pathSegment.ray.origin += 0.001f * pathSegment.ray.direction;

		// now check for intersections
		float t;
		glm::vec3 intersect_point;
		glm::vec3 normal;
		float t_min = FLT_MAX;
		int hit_geom_index = -1;
		bool outside = true;

		bool tmp_outside = true;
		glm::vec3 tmp_intersect;
		glm::vec3 tmp_normal;

		// naive parse through global geoms
		for (int i = 0; i < geoms_size; i++)
		{
			Geom & geom = geoms[i];

			if (geom.type == CUBE)
			{
				t = boxIntersectionTest(geom, pathSegment.ray, tmp_intersect, tmp_normal, tmp_outside);
			}
			else if (geom.type == SPHERE)
			{
				t = sphereIntersectionTest(geom, pathSegment.ray, tmp_intersect, tmp_normal, tmp_outside);
			}

			// Compute the minimum t from the intersection tests to determine what
			// scene geometry object was hit first.
			if (t > 0.0f && t_min > t)
			{
				t_min = t;
				hit_geom_index = i;
				intersect_point = tmp_intersect;
				normal = tmp_normal;
				outside = tmp_outside;
			}
		}

		if (hit_geom_index != -1)
		{
			//The ray hits something
			Material &mat = materials[geoms[hit_geom_index].materialid];
			if (mat.emittance > 0.0f)
				pathSegment.color *= (mat.color * mat.emittance);
		}
	}

}

// Add the current iteration's output to the overall image
__global__ void finalGather(int nPaths, glm::vec3 * image, PathSegment * iterationPaths)
{
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;

	if (index < nPaths)
	{
		PathSegment iterationPath = iterationPaths[index];
		image[iterationPath.pixelIndex] += iterationPath.color;
	}
}

__global__ void directGather(int nPaths, glm::vec3 * image, PathSegment * iterationPaths)
{
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;

	if (index < nPaths)
	{
		PathSegment iterationPath = iterationPaths[index];
		image[iterationPath.pixelIndex] *= iterationPath.color;
	}
}

__global__ void getActiveRays(int nPaths, int * active, int *inactive, const PathSegment * iterationPaths)
{
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;

	if (index < nPaths)
	{
		// NOTE: 1-index arrays so we don't compact element 0
		if (iterationPaths[index].remainingBounces > 0)
			active[index] = index + 1;
		else 
			inactive[index] = index + 1;
	}
}

__global__ void partitionRays(int nPathsPrev, int nPathsCompacted, int * active, int *inactive, PathSegment *out, const PathSegment *in)
{
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;

	if (index < nPathsCompacted)
	{
		out[index] = in[active[index] - 1];
	} else if (index < nPathsPrev) {
		out[index] = in[inactive[index - nPathsCompacted] - 1];
	}
}

/**
 * Wrapper for the __global__ call that sets up the kernel calls and does a ton
 * of memory management
 */
void pathtrace(uchar4 *pbo, int frame, int iter) {
    const int traceDepth = hst_scene->state.traceDepth;
    const Camera &cam = hst_scene->state.camera;
    const int pixelcount = cam.resolution.x * cam.resolution.y;
	const int materialcount = hst_scene->materials.size();

	// 2D block for generating ray from camera
    const dim3 blockSize2d(8, 8);
    const dim3 blocksPerGrid2d(
            (cam.resolution.x + blockSize2d.x - 1) / blockSize2d.x,
            (cam.resolution.y + blockSize2d.y - 1) / blockSize2d.y);

	// 1D block for path tracing
	const int blockSize1d = 128;

	int lightCount = 0;
	for (int i = 0; i < hst_scene->geoms.size(); i++) {
		if (hst_scene->materials[hst_scene->geoms[i].materialid].emittance > 0.0f)
			cudaMemcpy(&dev_lights[lightCount++], &dev_geoms[i], sizeof(Geom), cudaMemcpyDeviceToDevice);
	}

    ///////////////////////////////////////////////////////////////////////////

    // Recap:
    // * Initialize array of path rays (using rays that come out of the camera)
    //   * You can pass the Camera object to that kernel.
    //   * Each path ray must carry at minimum a (ray, color) pair,
    //   * where color starts as the multiplicative identity, white = (1, 1, 1).
    //   * This has already been done for you.
    // * For each depth:
    //   * Compute an intersection in the scene for each path ray.
    //     A very naive version of this has been implemented for you, but feel
    //     free to add more primitives and/or a better algorithm.
    //     Currently, intersection distance is recorded as a parametric distance,
    //     t, or a "distance along the ray." t = -1.0 indicates no intersection.
    //     * Color is attenuated (multiplied) by reflections off of any object
    //   * TODO: Stream compact away all of the terminated paths.
    //     You may use either your implementation or `thrust::remove_if` or its
    //     cousins.
    //     * Note that you can't really use a 2D kernel launch any more - switch
    //       to 1D.
    //   * TODO: Shade the rays that intersected something or didn't bottom out.
    //     That is, color the ray by performing a color computation according
    //     to the shader, then generate a new ray to continue the ray path.
    //     We recommend just updating the ray's PathSegment in place.
    //     Note that this step may come before or after stream compaction,
    //     since some shaders you write may also cause a path to terminate.
    // * Finally, add this iteration's results to the image. This has been done
    //   for you.

    // TODO: perform one iteration of path tracing

	int depth = 0;
	PathSegment* dev_path_end = dev_paths + pixelcount;
	int num_paths = dev_path_end - dev_paths;
	int num_paths_final = num_paths;

#if CACHE_FIRST_BOUNCE && !ANTIALIASING
	if (iter == 1) {
		generateRayFromCamera << <blocksPerGrid2d, blockSize2d >> >(cam, iter, traceDepth, dev_paths);
		checkCUDAError("generate camera ray");
		cudaMemcpy(dev_first_paths, dev_paths, num_paths*sizeof(PathSegment), cudaMemcpyDeviceToDevice);
	}
	else {
		cudaMemcpy(dev_paths, dev_first_paths, num_paths*sizeof(PathSegment), cudaMemcpyDeviceToDevice);
	}
#else
	generateRayFromCamera << <blocksPerGrid2d, blockSize2d >> >(cam, iter, traceDepth, dev_paths);
	checkCUDAError("generate camera ray");
#endif


	// --- PathSegment Tracing Stage ---
	// Shoot ray into scene, bounce between objects, push shading chunks

	bool iterationComplete = false;
	thrust::device_ptr<PathSegment> path_ptr = thrust::device_pointer_cast(dev_paths);
	thrust::device_ptr<ShadeableIntersection> inter_ptr = thrust::device_pointer_cast(dev_intersections);

	// Shared Compaction Test
	//int *dev_a, *dev_b;
	//cudaMalloc((void**)&dev_a, 8 * sizeof(int));
	//cudaMalloc((void**)&dev_b, 8 * sizeof(int));
	//int tmp[8] = { 1, 0, 2, 4, 0, 5, 6, 7};
	//cudaMemcpy(dev_a, tmp, 8 * sizeof(int), cudaMemcpyHostToDevice);
	//
	//int rtn = StreamCompaction::Efficient::compact_dev(8, dev_b, dev_a);
	//cudaMemcpy(tmp, dev_b, 8 * sizeof(int), cudaMemcpyDeviceToHost);

	//cudaFree(dev_a);
	//cudaFree(dev_b);

	while (!iterationComplete) {
		// clean shading chunks
		cudaMemset(dev_intersections, 0, pixelcount * sizeof(ShadeableIntersection));

		// tracing
		dim3 numblocksPathSegmentTracing = (num_paths + blockSize1d - 1) / blockSize1d;
		#if CACHE_FIRST_BOUNCE && !ANTIALIASING
		{
			if (iter > 1 && depth == 0) {
				cudaMemcpy(dev_intersections, dev_first_intersection, num_paths*sizeof(ShadeableIntersection), cudaMemcpyDeviceToDevice);
			} else {
				pathTraceOneBounce << <numblocksPathSegmentTracing, blockSize1d >> > (
					depth
					, num_paths
					, dev_paths
					, dev_geoms
					, hst_scene->geoms.size()
					, dev_intersections
					);

				checkCUDAError("trace one bounce");
				cudaDeviceSynchronize();

				if (iter == 1 && depth == 0)
					cudaMemcpy(dev_first_intersection, dev_intersections, num_paths*sizeof(ShadeableIntersection), cudaMemcpyDeviceToDevice);
			}
		}
		#else
		{
			pathTraceOneBounce << <numblocksPathSegmentTracing, blockSize1d >> > (
				depth
				, num_paths
				, dev_paths
				, dev_geoms
				, hst_scene->geoms.size()
				, dev_intersections
				);

			checkCUDAError("trace one bounce");
			cudaDeviceSynchronize();
		}
		#endif
		

		depth++;


		// TODO:
		// --- Shading Stage ---
		// Shade path segments based on intersections and generate new rays by
		// evaluating the BSDF.
		// Start off with just a big kernel that handles all the different
		// materials you have in the scenefile.
		// TODO: compare between directly shading the path segments and shading
		// path segments that have been reshuffled to be contiguous in memory.


		//std::chrono::time_point<std::chrono::system_clock> start, sort;
		//float sort_timer = 0;
		//start = std::chrono::system_clock::now();

		#if SORT_BY_MATERIAL
			thrust::stable_sort_by_key(inter_ptr, inter_ptr + num_paths, path_ptr, MaterialCmp());
		#endif

		shadeMaterialSimple << <numblocksPathSegmentTracing, blockSize1d >> > (
			iter,
			traceDepth,
			num_paths,
			dev_intersections,
			dev_paths,
			dev_materials
			);
		
#if CUSTOM_COMPACT
		// custom ray compaction
		cudaMemset(dev_active, 0, pixelcount * sizeof(int));
		cudaMemset(dev_inactive, 0, pixelcount * sizeof(int));
		getActiveRays << <numblocksPathSegmentTracing, blockSize1d >> > (num_paths, dev_active, dev_inactive, dev_paths);

		// compact remaining ray indices
		cudaDeviceProp prop;
		cudaGetDeviceProperties(&prop, 0);
		//printf("  threads: %i\n", prop.maxThreadsPerBlock);

//		int test_n = 512;
//		int buf[512];
//		cudaMemcpy(buf, dev_active, test_n*sizeof(int), cudaMemcpyDeviceToHost);
		int remainingPaths = StreamCompaction::Efficient::compact_dev(num_paths, dev_compacted, dev_active);
		checkCUDAError("compact active rays");
		cudaMemcpy(dev_active, dev_compacted, num_paths*sizeof(int), cudaMemcpyDeviceToDevice);
		//cudaMemcpy(buf, dev_active, test_n*sizeof(int), cudaMemcpyDeviceToHost);

		// compact completed ray indices
		StreamCompaction::Efficient::compact_dev(num_paths, dev_compacted, dev_inactive);
		cudaMemcpy(dev_inactive, dev_compacted, num_paths*sizeof(int), cudaMemcpyDeviceToDevice);
		checkCUDAError("compact inactive rays");


		// sort final path objects
		PathSegment *dev_out;
		cudaMalloc((void**)&dev_out, pixelcount*sizeof(PathSegment));
		
		partitionRays << <numblocksPathSegmentTracing, blockSize1d >> > (num_paths, remainingPaths, dev_active, dev_inactive, dev_out, dev_paths );
		cudaMemcpy(dev_paths, dev_out, num_paths*sizeof(PathSegment), cudaMemcpyDeviceToDevice);
		checkCUDAError("partition rays");


		num_paths = remainingPaths;
		cudaFree(dev_out);
#else
		// ray compaction with thrust
		thrust::partition(path_ptr, path_ptr + num_paths, TerminateRay());
		num_paths_final = num_paths;
		num_paths = thrust::count_if(path_ptr, path_ptr + num_paths, TerminateRay());

#endif 

		iterationComplete = (num_paths <= 0); // TODO: should be based off stream compaction results.
	}

	// Assemble this iteration and apply it to the image
	dim3 numBlocksPixels = (pixelcount + blockSize1d - 1) / blockSize1d;
	//finalGather << <numBlocksPixels, blockSize1d >> >(pixelcount, dev_image, dev_paths);

#if DIRECT_LIGHTING
	// Direct Lighting
	num_paths = pixelcount;
	//generateRayFromCamera << <blocksPerGrid2d, blockSize2d >> >(cam, iter, 8, dev_paths);

	//pathTraceOneBounce << <numBlocksPixels, blockSize1d >> > (
	//	0
	//	, num_paths
	//	, dev_paths
	//	, dev_geoms
	//	, hst_scene->geoms.size()
	//	, dev_intersections
	//	);
	//cudaDeviceSynchronize();
	//checkCUDAError("first object direct lighting");

	//shadeMaterialSimple << <numBlocksPixels, blockSize1d >> > (
	//	iter,
	//	traceDepth,
	//	num_paths,
	//	dev_intersections,
	//	dev_paths,
	//	dev_materials
	//	);
	//cudaDeviceSynchronize();
	//checkCUDAError("shade direct lighting");

	directLighting << <numBlocksPixels, blockSize1d >> > (
		iter,
		num_paths_final,
		dev_intersections,
		dev_paths,
		dev_materials,
		dev_geoms,
		hst_scene->geoms.size(),
		dev_lights,
		lightCount
		);
	checkCUDAError("find direct lighting");
#endif

	finalGather << <numBlocksPixels, blockSize1d >> >(pixelcount, dev_image, dev_paths);


    ///////////////////////////////////////////////////////////////////////////

    // Send results to OpenGL buffer for rendering
    sendImageToPBO<<<blocksPerGrid2d, blockSize2d>>>(pbo, cam.resolution, iter, dev_image);

    // Retrieve image from GPU
    cudaMemcpy(hst_scene->state.image.data(), dev_image,
            pixelcount * sizeof(glm::vec3), cudaMemcpyDeviceToHost);

    checkCUDAError("pathtrace");
}