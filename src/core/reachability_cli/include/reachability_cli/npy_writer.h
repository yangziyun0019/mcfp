// File: npy_writer.h
// Purpose: Declare utilities for exporting numeric arrays to the NumPy .npy format.
// Usage: Included by dataset-generation code paths that emit legacy array outputs.

#ifndef REACHABILITY_CLI__NPY_WRITER_H
#define REACHABILITY_CLI__NPY_WRITER_H

#include <cstddef>
#include <string>
#include <vector>

namespace reachability_cli
{

enum class NpyDataType
{
  kFloat32,
  kFloat64,
  kInt32,
  kInt64,
  kUInt8,
  kUInt32,
};

void writeNpy(const std::string& path, const void* data, const std::vector<size_t>& shape, NpyDataType dtype);

}  // namespace reachability_cli

#endif  // REACHABILITY_CLI__NPY_WRITER_H
