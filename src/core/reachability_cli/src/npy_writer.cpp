// File: npy_writer.cpp
// Purpose: Write NumPy .npy arrays used by the dataset-generation pipeline.
// Usage: Linked internally by reachability_cli dataset tools.

#include "reachability_cli/npy_writer.h"

#include <cstdint>
#include <fstream>
#include <sstream>
#include <stdexcept>

namespace reachability_cli
{
namespace
{
std::string dtypeToDescr(NpyDataType dtype)
{
  switch (dtype)
  {
    case NpyDataType::kFloat32:
      return "<f4";
    case NpyDataType::kFloat64:
      return "<f8";
    case NpyDataType::kInt32:
      return "<i4";
    case NpyDataType::kInt64:
      return "<i8";
    case NpyDataType::kUInt8:
      return "<u1";
    case NpyDataType::kUInt32:
      return "<u4";
  }
  return "<u1";
}

size_t dtypeSize(NpyDataType dtype)
{
  switch (dtype)
  {
    case NpyDataType::kFloat32:
      return 4;
    case NpyDataType::kFloat64:
      return 8;
    case NpyDataType::kInt32:
      return 4;
    case NpyDataType::kInt64:
      return 8;
    case NpyDataType::kUInt8:
      return 1;
    case NpyDataType::kUInt32:
      return 4;
  }
  return 1;
}

std::string shapeToString(const std::vector<size_t>& shape)
{
  std::ostringstream oss;
  oss << "(";
  for (size_t i = 0; i < shape.size(); ++i)
  {
    oss << shape[i];
    if (shape.size() == 1)
    {
      oss << ",";
      break;
    }
    if (i + 1 < shape.size())
    {
      oss << ", ";
    }
  }
  oss << ")";
  return oss.str();
}
}  // namespace

void writeNpy(const std::string& path, const void* data, const std::vector<size_t>& shape, NpyDataType dtype)
{
  std::ofstream out(path, std::ios::binary);
  if (!out)
  {
    throw std::runtime_error("Failed to open output file: " + path);
  }

  const std::string descr = dtypeToDescr(dtype);
  const std::string shape_str = shapeToString(shape);

  std::ostringstream header_stream;
  header_stream << "{'descr': '" << descr << "', 'fortran_order': False, 'shape': " << shape_str << ", }";
  std::string header = header_stream.str();

  const std::string magic = "\x93NUMPY";
  const unsigned char major = 1;
  const unsigned char minor = 0;

  std::size_t header_len = header.size() + 1;
  const std::size_t preamble = magic.size() + 2 + 2;
  const std::size_t padding = (16 - ((preamble + header_len) % 16)) % 16;
  header.append(padding, ' ');
  header.push_back('\n');
  header_len = header.size();

  out.write(magic.data(), static_cast<std::streamsize>(magic.size()));
  out.put(static_cast<char>(major));
  out.put(static_cast<char>(minor));

  const uint16_t header_len_u16 = static_cast<uint16_t>(header_len);
  out.write(reinterpret_cast<const char*>(&header_len_u16), sizeof(uint16_t));
  out.write(header.data(), static_cast<std::streamsize>(header.size()));

  if (!shape.empty())
  {
    std::size_t count = 1;
    for (size_t dim : shape)
    {
      count *= dim;
    }
    if (count > 0)
    {
      const std::size_t bytes = count * dtypeSize(dtype);
      out.write(reinterpret_cast<const char*>(data), static_cast<std::streamsize>(bytes));
    }
  }
}

}  // namespace reachability_cli
