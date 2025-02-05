# Edge Detection using NVIDIA CUDA kernels and cuDNN

**Capstone Project of the Coursera
[GPU Programming Specialization](https://www.coursera.org/specializations/gpu-programming)**

This project is based on my submission for the [coursera cuda at scale](https://github.com/alex-n-braun/coursera_cuda_at_scale) project.

| edges                              | generated from                   |
| ---------------------------------- | -------------------------------- |
| ![edges image](data/Lena_edge.png) | ![original image](data/Lena.png) |
|                                    |                                  |

An example video clip with a combined view of input- and output-data can be found on
[YouTube](https://youtu.be/U0ag6_rI0iA).

My **learning objectives** for this project:

- learn about cuDNN
- learn about CUDA graphs
- understand performance impact of various steps

## Table of Contents

- [Learning Objectives](#learning-objectives)
- [Key Concepts](#key-concepts)
- [Runtime Improvements](#runtime-improvements)
- [Project Overview and Build Instructions](#project-overview-and-build-instructions)
- [Final Words](#final-words)

## Learning Objectives

### Learning about cuDNN

The [previous project](https://github.com/alex-n-braun/coursera_cuda_at_scale) implemented an image filter for edge detection based on the Sobel operator using the _NVidia Performance Primitives_ (NPP) library. To learn more about cuDNN, I
attempted to replace NPP elements with cuDNN elements. Additionally, I wanted the
result to become more visually appealing, so I modified the filter to draw thicker
edges and inpaint them into the original image.

It turns out that cuDNN does not support some functions available in the
NPP lib -- surprise. Bitwise operations, integer-to-float conversions, and similar
functions can not be easily translated to cuDNN. This is fine since what I am
doing here is somewhat a misuse of cuDNN, as it is designed for implementing
neural networks, where differentiability is essential.

Of course, it was not difficult to implement the missing parts as plain CUDA kernels.

What I also learned: many functions in cuDNN were deprecated in cuDNN version 9.0.
There is an [overview](https://docs.nvidia.com/deeplearning/cudnn/backend/latest/api/overview.html)
explaining a modernization effort where a cuDNN Graph API was introduced in version 8.0. It also might be valuable to have a look into the [cuDNN release
notes](https://docs.nvidia.com/deeplearning/cudnn/backend/v9.7.0/release-notes.html)
in order to better understand what is going on.

### Learning about CUDA graphs

There are several recent developments in the CUDA space that are not covered in this course. I already mentioned the introduction of cuDNN graphs and the deprecation of several imperative cuDNN functions. Another interesting topic that I did not learn about in the Coursera GPU Programming Specialization is CUDA
graphs.

CUDA graphs allow for the recording of a predefined set of GPU operations that can then be executed repeatedly. There are several benefits: first, the graph manages the execution order. Each (sequential) call to a kernel creates a graph node, and CUDA can ultimately determine the dependency structure, even allowing for parallel execution of nodes that were recorded sequentially. Second, after the graph has been recorded, intermediate CPU-GPU interaction is removed: when executing the graph, everything runs on the GPU, which provides some performance benefits (as documented below).

Of course, there are also some drawbacks: once recorded, a graph is quite static. If the workflow includes (CPU-based) branching logic, this cannot be incorporated into the graph. Also, dynamic memory management is not possible: the necessary allocation steps must be performed before recording the graph.

### Performance Impact

I started this repo as a fork of my submission for the Coursera-course [CUDA at
scale for the Enterprise](https://github.com/alex-n-braun/coursera_cuda_at_scale).

As a first step, I altered the functionality to create a more visually appealing
representation of the edges, while replacing NPP with both plain CUDA kernels and
cuDNN calls.
The subsequent steps focused on reducing the runtime in various ways, including the use of CUDA graphs.

I conducted performance measurements by processing a [10s video
clip](https://youtu.be/U0ag6_rI0iA). (The clip presented on YouTube is post-processed into a combined view of input- and output-data). The achieved
performance improvements are quite impressive: starting from roughly 42ms
processing time per video frame with a naive implementation, I was finally able
to achieve a total runtime of roughly 20ms, which is an improvement of more than
50%. This measurement includes file io and memory transfer between CPU and GPU.

If we consider the runtime of GPU operations alone, the runtime dropped from
roughly 33ms to less than 6ms, which represents a reduction of 82%! Still, I am
convinced that a seasoned CUDA developer (which I am not yet) will be able to
optimize this further.

I will discuss each step in more detail below in the [Runtime Improvements](#runtime-improvements) section.

## Key Concepts

The application performs multiple steps:

1. Loading the data from disk (either image, or video)
2. Manipulate image or single frames from the video
3. Store manipulated data back to disk, either as an image, or as a video file.

The image processing makes use of several CUDA kernels and cuDNN convolutions:

1. Convert `uint8` quantized image to `float`
   [`convertUint8ToFloat`](./include/image_manip.hpp) (One should consider
   implementing the filter completely in `int` quantization for potential performance
   improvements.)
2. Convert RGB image to grayscale using a cuDNN convolution operator
   ([`m_conv_to_grayscale`](./include/filter.hpp#L30))
3. Apply a sobel operator to detect edges: cuDNN convolution to find horizontal
   and vertical edges [`m_conv_edges`](./include/filter.hpp#L32),
   [`pointwiseAbs`](./include/image_manip.hpp) ensures all values are positive,
   and another cuDNN conv to reduce the 2-channel image with vertical and horizontal
   edges to a 1-channel image representing all edges
   [`m_conv_reduce_2d_to_1d`](./include/filter.hpp#L39)
4. Looping n times, applying another cuDNN convolution to smooth the edges
   [`m_conv_smooth`](./include/filter.hpp#L40)
   while limiting the intensity using [`pointwiseMin`](./include/image_manip.hpp)
5. Applying a convolution [`m_conv_delete`](./include/filter.hpp#L46): the
   convolution kernel is designed to reduce intensity in areas without intensity
   variations. It is essentially a variant of an edge detection kernel. Afterward,
   another [`pointwiseMin`](./include/image_manip.hpp) introduces an intensity
   cutoff.
6. A convolution [`m_conv_broadcast_to_4_channels`](./include/filter.hpp#L31) transforms
   the 1-channel grayscale edge image into a 4-channel RGBA image,
   [`pointwiseHalo`](./include/image_manip.hpp) combines the edge image with
   the initial color image, and [`setChannel`](./include/image_manip.hpp) sets the
   intensity of the alpha channel to the maximum value.
7. Finally, convert the floating point image back to `uint8`
   [`convertFloatToUint8`](./include/image_manip.hpp).

The filter's code can be found in `Filter::runFilterOnGpu` in
[`filter.hpp`](./include/filter.hpp#L80). To simplify the relatively verbous
implementation of a cuDNN convolution, I introduced the class
[`Convolution`](./include/convolution.hpp). The CUDA kernels underlying the
various functions like [`convertUint8ToFloat`](./include/image_manip.hpp),
[`convertFloatToUint8`](./include/image_manip.hpp),
[`m_conv_to_grayscale`](./include/filter.hpp#L30),
[`pointwiseAbs`](./include/image_manip.hpp),
[`pointwiseMin`](./include/image_manip.hpp),
[`pointwiseHalo`](./include/image_manip.hpp), and
[`setChannel`](./include/image_manip.hpp), are implemented in [`cuda_kernels.cu`](./src/cuda_kernels.cu)

The project is structured into several additional files. The class
[`Cli`](./include/cli.hpp) manages the command line interface. The class
[`CudaGraph`](./include/cuda_graph.hpp) abstracts the capturing and the execution
of a CUDA graph. [`GpuBlob`](./include/gpu_blob.hpp) handles GPU memory
management, including allocation and deallocation, as well as transfer between CPU
and GPU at a low level. It is used by the [`ImageGPU`](./include/types.hpp#L51)
class and the [`Kernel`](./include/types.hpp#L11) class.
[`GpuSession`](./include/gpu_session.hpp) manages session handles. Several helper
functions have been from the NVidia
[cuda samples](https://github.com/NVIDIA/cuda-samples).
[`io.hpp`](./include/io.hpp) handles file i/o, and, last but not least,
[`Timer`](./include/timer.hpp) provides a simple class for performing runtime
measurements.

The main program is implemented in [`edgeDetection.cpp`](./src/edgeDetection.cpp).
The discussed CUDA graph offers no advantage over imperative execution
when processing only a single image. Therefore, I implemented a graph only in
[`processVideo`](./src/edgeDetection.cpp#L64), which reads from a video
clip, applies the graph frame-by-frame, and writes the frames back into a new video
clip.

## Runtime Improvements

In the following sections, I compare different implementation stages of the edge detection program. My focus was on learning about CUDA's behavior with respect
to runtime. For
[each PR](https://github.com/alex-n-braun/coursera_cuda_advanced_libraries/pulls?q=is%3Apr+)
I carried out a simple runtime measurement of different parts of the program,
which are presented in the tables below.

### Reference Setup

As stated before, the project is based on a
[fork](https://github.com/alex-n-braun/coursera_cuda_at_scale). With the
[first PR](https://github.com/alex-n-braun/coursera_cuda_advanced_libraries/pull/1),
I introduced the enhanced functionality of edge inpainting into the original
image, while removing CUDA NPP and replacing the relevant
functionalities with CUDA kernels and cuDNN convolutions the same time. This somewhat **naive implementation** does not account for any bottlenecks.

The runtime measurement for a 10s video clip (1280x720, 25fps) produced the
following table: **(elapsed time in nanoseconds)**

|                  | Total [ns]  | per frame [ns] |
| ---------------- | ----------- | -------------- |
| incl. io         | 10675647389 | 42532459       |
| excl. io         | 9140242967  | 36415310       |
| gpu              | 8443221916  | 33638334       |
| w/o conv. to int | 8401399737  | 33471712       |

The two columns indicate the total runtime and the runtime per frame.
The rows represent:

- The runtime including file i/o, meaning the time required to read a frame from a
  file, process the frame, and store it back to a file.
- The runtime excluding file i/o, but including memory transfers between CPU and
  GPU.
- The runtime of GPU operations only, which includes data conversion steps between
  integer-quantized images and a float representation.
- The runtime of the GPU operations, excluding data conversion steps.

The above table serves as the reference measurement. The following sections outline
optimization steps that aim to reduce the runtime.

### Reduction in the Number of Memory Allocations

The [next relevant PR](https://github.com/alex-n-braun/coursera_cuda_advanced_libraries/pull/4)
introduces two changes; therefore I will refer to individual commits.

When performing image manipulation tasks on the GPU, it is necessary to allocate
memory. The naive implementation handles this inefficiently: for each call
to the filter function,
[temporary memory is allocated](https://github.com/alex-n-braun/coursera_cuda_advanced_libraries/pull/4/commits/20867d9817edfb78ae342d7e6af7d7a5f2514d12#diff-5a70dea16dccea0d8b3f2710191bb64be65cbeba4e2914f1e3ce5df155073480L103),
and before exiting the
function, it is freed again. The commit moves the temporary memory to
[mutable
members](https://github.com/alex-n-braun/coursera_cuda_advanced_libraries/pull/4/commits/20867d9817edfb78ae342d7e6af7d7a5f2514d12#diff-5a70dea16dccea0d8b3f2710191bb64be65cbeba4e2914f1e3ce5df155073480R116)
of the `Filter` class.

This results in a runtime improvement of approximately 5% for GPU-based
computations, excluding float<=>int conversions.

|                  | Total [ns] | per frame [ns] |
| ---------------- | ---------- | -------------- |
| incl. io         | 9823949656 | 39139241       |
| excl. io         | 8311901920 | 33115147       |
| gpu              | 7938228023 | 31626406       |
| w/o conv. to int | 7930628447 | 31596129       |

### Creation and Destruction of Handles and Descriptors

At a higher level, it is necessary to manage various handles and descriptors, which
are created and destroyed using functions like
[`cudnnCreate`](https://docs.nvidia.com/deeplearning/cudnn/backend/latest/api/cudnn-graph-library.html#cudnncreate) /
[`cudnnDestroy`](https://docs.nvidia.com/deeplearning/cudnn/backend/latest/api/cudnn-graph-library.html#cudnndestroy) ,
[`cudnnCreateConvolutionDescriptor`](https://docs.nvidia.com/deeplearning/cudnn/backend/latest/api/cudnn-cnn-library.html#cudnncreateconvolutiondescriptor) /
[`cudnnDestroyConvolutionDescriptor`](https://docs.nvidia.com/deeplearning/cudnn/backend/latest/api/cudnn-cnn-library.html#cudnndestroyconvolutiondescriptor)
etc. Again, the naive implementation handles this on a frame-by-frame level. It
turns out that this introduces significant overhead.

The [relevant commit](https://github.com/alex-n-braun/coursera_cuda_advanced_libraries/pull/4/commits/6a9dc076af0cefb6c5852e8159c76b5499228c70)
moves the creation and destruction of the descriptors directly related to the
convolution into the constructor and destructor of the class. Since the cuDNN
handle is needed by all cuDNN operations, I introduce the
[`GpuSession`](https://github.com/alex-n-braun/coursera_cuda_advanced_libraries/pull/4/commits/6a9dc076af0cefb6c5852e8159c76b5499228c70#diff-aa2864769634e4c25e08571438c2e9ed52d5efd14f76ec15f5a0791156aede9cR19)
class.

**This results additional frame-processing runtime improvement of approximately 81%** (wow!).

|                  | Total [ns] | per frame [ns] |
| ---------------- | ---------- | -------------- |
| incl. io         | 5260741121 | 20959127       |
| excl. io         | 2015754933 | 8030896        |
| gpu              | 1509760936 | 6014983        |
| w/o conv. to int | 1488816208 | 5931538        |

Clearly, minimizing unnecessary cuDNN handle creation has a major impact on
performance. Further investigation into the runtime impact of creating and destroying `cudnnHandle_t` and other objects, such as `cudnnTensorDescriptor_t` could provide additional insights.

### Caching

It is possible to further
[reduce operations](https://github.com/alex-n-braun/coursera_cuda_advanced_libraries/pull/5) that may be
redundant, such as repeatedly setting image width and height , even though these
should remain constant for an entire video clip. This results in an additional
runtime improvement of approximately 2–3%.

|                  | Total [ns]     | per frame [ns] |
| ---------------- | -------------- | -------------- |
| incl. io         | 4944791394     | 19700364       |
| excl. io         | 1908645536     | 7604165        |
| **gpu**          | **1460924323** | **5820415**    |
| w/o conv. to int | 1448427256     | 5770626        |

The row **gpu** is marked bold as it serves as the reference for the next step, as
explained below.

### CUDA graph

The [final step](https://github.com/alex-n-braun/coursera_cuda_advanced_libraries/pull/6)
in optimizing the GPU part is to add all relevant operations to a CUDA _graph_, which is then replayed once per frame during the video processing. This results in
a further runtime improvement of roughly 4-5%.
Note that it is no longer possible to differentiate between individual steps
running inside the graph (such as excluding type casts, as done in the previous
performance measurement). Doing so would require adding performance measurements _inside_ the graph, which I do not (yet) know how to implement. However, running
the graph _including_ the type casts still takes less time than the previous imperative implementation, even when _excluding_ the type casts in the runtime
measurement.

Additionally, since CPU-based branching cannot be recorded within the graph, the
caching mechanism for setting image width and height had to be removed. It is now
assumed that the image resolution remains fixed throughout the entire run.

|          | Total [ns]     | per frame [ns] |
| -------- | -------------- | -------------- |
| incl. io | 4778412190     | 19037498       |
| excl. io | 1795590019     | 7153745        |
| **gpu**  | **1391767538** | **5544890**    |

### What else?

There is much more that could be optimized:

- Use of pinned memory
- Implementation using `int` quantization
- Potential use of asynchronous operations
- Inclusion of mem copy operations into the cuda graph
- Batch processing
- Multithreading on CPU to optimize I/O performance
- ...

However, since I already have spent quite some time with this project, I will stop here for now.

## Project Overview and Build Instructions

### Code Organization

`bin/`
After building using make, this folder will hold the executable `edgeDetection` together with a number of object files.

`data/`
This folder holds example data, an image [`Lena.png`](./data/Lena.png). By default, the output is stored in the same folder.

`include/`
Holds header files.

`src/`
Holds the source code of the project

`.clang-format`, `.clang-tidy`
Config files for setting up clang tools that can be used for formatting and
cleaning the code.

`README.md`
This file describes the project. It holds human-readable instructions for building and executing the code.

`Makefile`
This file contains the instructions for building the project using the make utility. It specifies the dependencies, compilation flags, and the target executable to be generated.

### Supported OSes

The project was testet on Ubuntu 24.04.

### Supported CPU Architecture

The project was tested on x86_64.

### CUDA APIs involved

NVIDIA CUDA Deep Neural Network library (cuDNN)

### Dependencies needed to build/run

- [FreeImage](https://freeimage.sourceforge.io/) On Ubuntu, `apt install libfreeimage-dev`
- [cuDNN](https://docs.nvidia.com/deeplearning/cudnn/latest/) Need to [install NVidia cuDNN Backend](https://docs.nvidia.com/deeplearning/cudnn/installation/latest/linux.html)
- [opencv](https://opencv.org/) `apt install libopencv-dev`

For using tools like clang-format and clang-tidy, you need to install separate
packages

### Prerequisites

Download and install the [CUDA Toolkit](https://developer.nvidia.com/cuda-downloads) for your corresponding platform. The project was tested with CUDA 12.5.
Make sure the dependencies mentioned in [Dependencies]() section above are installed.

### Tools

- `bear` for creating a `compile_commands.json` file. This can be used for clangd integration in vscode.
- `clang-tidy`, `clang-format` as linter and for code formatting

### Build and Run

The project has been tested on Ubuntu 24.04. There is a [`Makefile`](./Makefile), therefore the project can be built using

```
$ make all
```

### Running the Program

You can run the program using the following command:

```bash
make run
```

This command will execute the compiled binary, applying the edge filter on the example input image (Lena.png), and save the result as Lena_edge.png in the data/ directory.

If you wish to run the binary directly with custom input/output files, you can use:

```bash
./bin/edgeDetection --input data/Lena.png --output data/Lena_edges.png
```

You can also process a video file. The i/o is implemented using opencv:

```bash
./bin/edgeDetection --input some_input.mp4 --output edges_video.mp4
```

Cleaning Up: To clean up the compiled binaries and other generated files, run:

```bash
make clean
```

This will remove all files in the bin/ directory.

## Final Words

This project has deepened my understanding of GPU programming, CUDA, and performance optimization. The Coursera GPU Programming Specialization provided a solid foundation, covering:

- Concurrent programming with GPUs
- Parallel programming using CUDA
- Scaling CUDA applications
- Utilizing advanced CUDA libraries

However, several key areas were not covered, particularly:

- Modern CUDA Features: Advancements like CUDA Graphs for optimizing execution.
- Tool Landscape:
   - Profiling: NVIDIA Nsight Systems & Compute
   - Benchmarking: Performance assessment tools
   - Debugging: Advanced GPU debugging utilities

While this project and coursework offered a strong start, ongoing learning and hands-on experimentation will be essential for mastering modern GPU development.
