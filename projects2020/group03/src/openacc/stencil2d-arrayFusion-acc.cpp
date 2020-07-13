#include <omp.h>

#include <cassert>
#include <chrono>
#include <fstream>
#include <iostream>

#ifdef CRAYPAT
#include "pat_api.h"
#endif
#include "../utils.h"

namespace {

// overloaded function with cstd arrays
// #pragma acc routine seq
void inline updateHalo(double *inField, int32_t xsize, int32_t ysize,
                       int32_t zsize, int32_t halosize) {
  std::size_t xMin = halosize;
  std::size_t xMax = xsize - halosize;
  std::size_t yMin = halosize;
  std::size_t yMax = ysize - halosize;
  std::size_t zMin = 0;
  std::size_t zMax = zsize;

  const int xInterior = xMax - xMin;
  const int yInterior = yMax - yMin;

  std::size_t sizeOf3DField = (xsize) * (ysize) * (zsize);

  // todo : do you need the copy in?
  // #pragma acc data copyin(inField [0:sizeOf3DField])
  {
#pragma acc enter data copyin(inField [0:sizeOf3DField])
    // bottom edge (without corners)
#pragma acc parallel present(inField) async(1)
#pragma acc loop independent gang worker vector collapse(3)
    for (std::size_t k = 0; k < zMin; ++k) {
      for (std::size_t j = 0; j < yMin; ++j) {
        for (std::size_t i = xMin; i < xMax; ++i) {
          std::size_t ijk = i + (j * xsize) + (k * xsize * ysize);
          std::size_t ijp1k =
              i + ((j + yInterior) * xsize) + (k * xsize * ysize);

          inField[ijk] = inField[ijp1k];
          // inField[k][j][i] = inField[k][j + yInterior][i];
        }
      }
    }

    // top edge (without corners)
#pragma acc parallel present(inField) async(2)
#pragma acc loop independent gang worker vector collapse(3)
    for (std::size_t k = 0; k < zMin; ++k) {
      for (std::size_t j = yMax; j < ysize; ++j) {
        for (std::size_t i = xMin; i < xMax; ++i) {
          std::size_t ijk = i + (j * xsize) + (k * xsize * ysize);
          std::size_t ijm1k =
              i + ((j - yInterior) * xsize) + (k * xsize * ysize);

          inField[ijk] = inField[ijm1k];
          // inField[k][j][i] = inField[k][j - yInterior][i];
        }
      }
    }

    // left edge (including corners)
#pragma acc parallel present(inField) async(3)
#pragma acc loop independent gang worker vector collapse(3)
    for (std::size_t k = 0; k < zMin; ++k) {
      for (std::size_t j = yMin; j < yMax; ++j) {
        for (std::size_t i = 0; i < xMin; ++i) {
          std::size_t ijk = i + (j * xsize) + (k * xsize * ysize);
          std::size_t ip1jk = i + xInterior + (j * xsize) + (k * xsize * ysize);

          inField[ijk] = inField[ip1jk];
          // inField[k][j][i] = inField(i + xInterior, j, k);
        }
      }
    }

    // right edge (including corners)
#pragma acc parallel present(inField) async(4)
#pragma acc loop independent gang worker vector collapse(3)
    for (std::size_t k = 0; k < zMin; ++k) {
      for (std::size_t j = yMin; j < yMax; ++j) {
        for (std::size_t i = xMax; i < xsize; ++i) {
          std::size_t ijk = i + (j * xsize) + (k * xsize * ysize);
          std::size_t im1jk = i - xInterior + (j * xsize) + (k * xsize * ysize);

          inField[ijk] = inField[im1jk];
          // inField[k][j][i] = inField[i - xInterior][j][k];
        }
      }
    }
  }
#pragma acc exit data copyout(inField [0:sizeOf3DField])

} // namespace

// overloaded function with cstd arrays
void apply_diffusion(double *inField, double *outField, double alpha,
                     unsigned numIter, int x, int y, int z, int halo) {

  // TODO : temp array or restrict inField to avoid aliasing?
  std::size_t xsize = x;
  std::size_t xMin = halo;
  std::size_t xMax = xsize - halo;

  std::size_t ysize = y;
  std::size_t yMin = halo;
  std::size_t yMax = ysize - halo;

  std::size_t zMin = 0;
  std::size_t zMax = z;

  std::size_t sizeOf3DField = (xsize) * (ysize)*z;

  // #pragma acc data copyin(inField [0:sizeOf3DField])                             \
  //     copy(outField [0:sizeOf3DField])

#pragma acc enter data \
	copyin(inField [0:sizeOf3DField]) \
	copyin(outField [0:sizeOf3DField])
  for (std::size_t iter = 0; iter < numIter; ++iter) {

    // TODO : make this an acc routine
    // todo : turn this into a cuda kernel and call from acc???
    // #pragma acc parallel present(inField)
    updateHalo(inField, xsize, ysize, z, halo);
    // todo : if this is not on the gpu, the gpu copy needs to be updated?
    // # pragma acc update device(inField[0:n])

#pragma acc parallel present(inField)
#pragma acc loop gang worker vector collapse(3)
    for (std::size_t k = 0; k < zMax; ++k) {
      for (std::size_t j = yMin; j < yMax; ++j) {
        for (std::size_t i = xMin; i < xMax; ++i) {
          std::size_t ijk = i + (j * xsize) + (k * xsize * ysize);
          std::size_t ip1jk = (i + 1) + (j * xsize) + (k * xsize * ysize);
          std::size_t im1jk = (i - 1) + (j * xsize) + (k * xsize * ysize);
          std::size_t ijp1k = i + ((j + 1) * xsize) + (k * xsize * ysize);
          std::size_t ijm1k = i + ((j - 1) * xsize) + (k * xsize * ysize);
          std::size_t im1jm1k = i - 1 + ((j - 1) * xsize) + k * (xsize * ysize);
          std::size_t im1jp1k = i - 1 + ((j + 1) * xsize) + k * (xsize * ysize);
          std::size_t ip1jm1k = i + 1 + ((j - 1) * xsize) + k * (xsize * ysize);
          std::size_t ip1jp1k = i + 1 + ((j + 1) * xsize) + k * (xsize * ysize);
          std::size_t im2jk = (i - 2) + (j * xsize) + k * (xsize * ysize);
          std::size_t ip2jk = (i + 2) + (j * xsize) + k * (xsize * ysize);
          std::size_t ijm2k = i + ((j - 2) * xsize) + k * (xsize * ysize);
          std::size_t ijp2k = i + ((j + 2) * xsize) + k * (xsize * ysize);

          double partial_laplap =
              // 20*inField[ijk] -
              -8 * (inField[im1jk] + inField[ip1jk] + inField[ijm1k] +
                    inField[ijp1k]) +
              2 * (inField[im1jm1k] + inField[ip1jm1k] + inField[im1jp1k] +
                   inField[ip1jp1k]) +
              1 * (inField[im2jk] + inField[ip2jk] + inField[ijm2k] +
                   inField[ijp2k]);

          // TODO : check if independent
          outField[ijk] =
              (1 - 20 * alpha) * inField[ijk] - alpha * partial_laplap;
        }
      }
    }
    if (iter = numIter - 1) {
#pragma acc parallel present(inField, outField)
#pragma acc loop independent gang worker vector collapse(3)
      for (std::size_t k = 0; k < zMax; ++k) {
        for (std::size_t j = yMin; j < yMax; ++j) {
          for (std::size_t i = xMin; i < xMax; ++i) {
            std::size_t ijk = i + (j * xsize) + (k * xsize * ysize);
            inField[ijk] = outField[ijk];
          }
        }
      }
    }
  }
#pragma acc exit data \
	copyout(outField[0:sizeOf3DField]) \
	delete(inField[0:sizeOf3DField])
} // namespace

void reportTime(const Storage3D<double> &storage, int nIter, double diff) {
  std::cout << "# ranks nx ny ny nz num_iter time\ndata = np.array( [ \\\n";
  int size;
#pragma omp parallel
  {
#pragma omp master
    { size = omp_get_num_threads(); }
  }
  std::cout << "[ " << size << ", " << storage.xMax() - storage.xMin() << ", "
            << storage.yMax() - storage.yMin() << ", " << storage.zMax() << ", "
            << nIter << ", " << diff << "],\n";
  std::cout << "] )" << std::endl;
}
} // namespace

int main(int argc, char const *argv[]) {
#ifdef CRAYPAT
  PAT_record(PAT_STATE_OFF);
#endif
  int x = atoi(argv[2]);
  int y = atoi(argv[4]);
  int z = atoi(argv[6]);
  int iter = atoi(argv[8]);
  int nHalo = 3;
  assert(x > 0 && y > 0 && z > 0 && iter > 0);

  std::size_t zsize, xsize, ysize;
  xsize = x + 2 * nHalo;
  ysize = y + 2 * nHalo;
  zsize = z;

  Storage3D<double> input_3D(x, y, z, nHalo);
  Storage3D<double> output_3D(x, y, z, nHalo);

  input_3D.initialize();
  output_3D.initialize();

  std::size_t sizeOf3DField = (xsize) * (ysize) * (zsize);
  double *input = new double[sizeOf3DField];
  double *output = new double[sizeOf3DField];

  // zero initialize the newly allocated array
  for (std::size_t k = 0; k < zsize; ++k) {
    for (std::size_t j = 0; j < ysize; ++j) {
      for (std::size_t i = 0; i < xsize; ++i) {
        std::size_t index1D = i + (j * xsize) + (k * xsize * ysize);
        input[index1D] = 0;
        output[index1D] = 0;
      }
    }
  }

  // initial condition
  for (std::size_t k = zsize / 4.0; k < 3 * zsize / 4.0; ++k) {
    for (std::size_t j = nHalo + xsize / 4.; j < nHalo + 3. / 4. * xsize; ++j) {
      for (std::size_t i = nHalo + xsize / 4.; i < nHalo + 3. / 4. * xsize;
           ++i) {
        input[k * (ysize * xsize) + j * xsize + i] = 1;
        output[k * (ysize * xsize) + j * xsize + i] = 1;
      }
    }
  }

  double alpha = 1. / 32.;

  // copy input to input_3D for writing to file
  std::ofstream fout;
  fout.open("in_field.dat", std::ios::binary | std::ofstream::trunc);
  input_3D.writeFile(fout);
  fout.close();

#ifdef CRAYPAT
  PAT_record(PAT_STATE_ON);
#endif
  auto start = std::chrono::steady_clock::now();

  // apply_diffusion(input_3D, output_3D, alpha, iter, x, y, z, nHalo);
  apply_diffusion(input, output, alpha, iter, xsize, ysize, zsize, nHalo);

  auto end = std::chrono::steady_clock::now();
#ifdef CRAYPAT
  PAT_record(PAT_STATE_OFF);
#endif

  // updateHalo(output_3D);
  updateHalo(output, xsize, ysize, zsize, nHalo);

  // copy output array to output_3D for writing to file
  for (std::size_t k = 0; k < zsize; ++k) {
    for (std::size_t j = 0; j < ysize; ++j) {
      for (std::size_t i = 0; i < xsize; ++i) {
        std::size_t index1D = i + (j * xsize) + (k * xsize * ysize);
        output_3D(i, j, k) = output[index1D];
      }
    }
  }

  fout.open("out_field.dat", std::ios::binary | std::ofstream::trunc);
  output_3D.writeFile(fout);
  fout.close();

  auto diff = end - start;
  double timeDiff =
      std::chrono::duration<double, std::milli>(diff).count() / 1000.;
  reportTime(output_3D, iter, timeDiff);

  delete[] input;
  delete[] output;

  return 0;
}
